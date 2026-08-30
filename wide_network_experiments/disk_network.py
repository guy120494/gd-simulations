"""
2-layer ReLU network for d-dimensional inputs.

Architecture:  f(x) = v^T ReLU(Wx + b)
  W ∈ R^{k×d}  — first-layer weights
  b ∈ R^k      — first-layer biases
  v ∈ R^k      — output weights

All three parameter arrays are trained.
"""

import warnings

import numpy as np


def forward(W: np.ndarray, b: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute network outputs for a batch of inputs.

    Parameters
    ----------
    W : (k, d)
    b : (k,)
    v : (k,)
    X : (n, d)

    Returns
    -------
    (n,) array of scalar outputs
    """
    return np.maximum(0.0, X @ W.T + b) @ v


def exp_loss(y: np.ndarray, preds: np.ndarray) -> float:
    """Mean exponential loss: (1/n) Σ exp(-y_i · f(x_i))."""
    return float(np.mean(np.exp(-y * preds)))


def gradients(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """
    Gradients of the exponential loss w.r.t. W, b, v.

    Returns
    -------
    grad_W : (k, d)
    grad_b : (k,)
    grad_v : (k,)
    """
    n = len(y)
    pre = X @ W.T + b                                     # (n, k) pre-activations
    act = np.maximum(0.0, pre)                            # (n, k) post-activations
    out = act @ v                                         # (n,)   network outputs

    dL_dout = -(1.0 / n) * y * np.exp(-y * out)          # (n,)
    indicator = (pre > 0).astype(float)                   # (n, k) subgradient of ReLU

    dL_dpre = dL_dout[:, None] * v[None, :] * indicator  # (n, k)
    grad_W = dL_dpre.T @ X                               # (k, d)
    grad_b = dL_dpre.sum(axis=0)                         # (k,)
    grad_v = act.T @ dL_dout                             # (k,)

    return grad_W, grad_b, grad_v


def _distance_to_boundary_piece(
    x_i: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    a: np.ndarray,
    c: float,
    a_sq: float,
) -> float:
    """
    Exact distance from x_i to the boundary piece H_σ ∩ R_σ, solved as a QP:

        min  ½‖x − x_i‖²
        s.t. aᵀx + c = 0              (x lies on the boundary hyperplane H_σ)
             (Wx + b)_j ≥ 0  ∀j: σ_j = 1   (active-neuron region constraints)
             (Wx + b)_j ≤ 0  ∀j: σ_j = 0   (inactive-neuron region constraints)

    The foot of the perpendicular onto H_σ is used as the warm start; when it
    already satisfies R_σ the solver returns immediately.

    Returns np.inf if the intersection H_σ ∩ R_σ is empty or the solver fails.
    """
    from scipy.optimize import minimize

    k = len(b)

    # Warm start: project x_i onto H_σ (always feasible for the equality constraint)
    x0 = x_i - ((float(a @ x_i) + c) / a_sq) * a

    constraints = [
        {"type": "eq", "fun": lambda x: float(a @ x) + c, "jac": lambda x: a},
    ]
    for j in range(k):
        if sigma[j]:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, j=j: float(W[j] @ x) + b[j],
                "jac": lambda x, j=j: W[j],
            })
        else:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, j=j: -(float(W[j] @ x) + b[j]),
                "jac": lambda x, j=j: -W[j],
            })

    result = minimize(
        lambda x: 0.5 * float(np.dot(x - x_i, x - x_i)),
        x0,
        jac=lambda x: x - x_i,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )

    x_opt = result.x

    # Verify the solution actually satisfies all constraints (catches infeasible cases
    # where SLSQP converges to a point that violates the feasible set).
    tol = 1e-5
    if abs(float(a @ x_opt) + c) > tol:
        return np.inf

    pre_opt = W @ x_opt + b
    if sigma.any() and np.max(-pre_opt[sigma]) > tol:
        return np.inf
    if (~sigma).any() and np.max(pre_opt[~sigma]) > tol:
        return np.inf

    return float(np.linalg.norm(x_opt - x_i))


def compute_boundary_distances(
    W: np.ndarray, b: np.ndarray, v: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Exact distance from each point in X to the decision boundary {f = 0}.

    The network is piecewise linear. For each activation pattern σ ∈ {0,1}^k
    the network reduces to a linear function in its region:

        f_σ(x) = a_σᵀx + c_σ,   a_σ = Wᵀ(v ⊙ σ),   c_σ = b · (v ⊙ σ)

    The boundary piece in that region is H_σ ∩ R_σ, where
        H_σ = {x : a_σᵀx + c_σ = 0}
        R_σ = {x : (Wx + b)_j ≥ 0 iff σ_j = 1}

    For each point x_i and each pattern σ, the distance to H_σ ∩ R_σ is found
    by solving a convex QP (see _distance_to_boundary_piece).  As a fast path,
    when the foot of the perpendicular from x_i onto H_σ already lies in R_σ,
    the closed-form distance |a_σᵀx_i + c_σ| / ‖a_σ‖ is used directly.

    The reported distance for each point is the minimum over all σ.

    Returns
    -------
    (n,) array; np.inf for any point with no reachable boundary piece.

    Note: runtime is O(2^k · n · QP_solve) — a warning is issued for k > 20.
    """
    k = W.shape[0]
    if k > 20:
        warnings.warn(
            f"k={k}: enumerating 2^k = {2**k:,} activation patterns is expensive.",
            stacklevel=2,
        )

    n = X.shape[0]
    distances = np.full(n, np.inf)

    for sigma_int in range(1, 1 << k):  # skip all-zeros: gives a = 0, no boundary
        sigma = np.array([(sigma_int >> j) & 1 for j in range(k)], dtype=bool)

        a = W.T @ (v * sigma)       # (d,) — normal to the boundary hyperplane
        c = float(b @ (v * sigma))  # scalar offset

        a_sq = float(a @ a)
        if a_sq < 1e-14:            # f_σ is constant in this region → no boundary piece
            continue

        # --- Fast path: closed-form distance when foot lies in R_σ ---
        sd = X @ a + c                                   # (n,) signed distances × ‖a‖
        feet = X - (sd / a_sq)[:, None] * a[None, :]    # (n, d)

        pre_at_feet = feet @ W.T + b                     # (n, k)
        foot_valid = np.ones(n, dtype=bool)
        if sigma.any():
            foot_valid &= (pre_at_feet[:, sigma] >= -1e-9).all(axis=1)
        if (~sigma).any():
            foot_valid &= (pre_at_feet[:, ~sigma] <= 1e-9).all(axis=1)

        foot_dist = np.abs(sd) / np.sqrt(a_sq)
        distances[foot_valid] = np.minimum(distances[foot_valid], foot_dist[foot_valid])

        # --- Slow path: QP for points whose foot fell outside R_σ ---
        for i in np.where(~foot_valid)[0]:
            d_qp = _distance_to_boundary_piece(X[i], W, b, sigma, a, c, a_sq)
            if d_qp < distances[i]:
                distances[i] = d_qp

    return distances
