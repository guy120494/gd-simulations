"""
Training loop for the 2-disk experiment.

Supports GD and Adam optimizers. The caller supplies an optional
`on_checkpoint` callback that is invoked at every reporting step,
keeping the training logic decoupled from result collection.
"""

from typing import Any, Dict, Optional

import numpy as np

from .disk_network import compute_boundary_distances, forward, exp_loss, gradients


def _adam_step(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    grad_W: np.ndarray,
    grad_b: np.ndarray,
    grad_v: np.ndarray,
    state: Dict[str, Any],
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
):
    state["t"] += 1
    t = state["t"]
    state["m_W"]  = beta1 * state["m_W"]  + (1 - beta1) * grad_W
    state["sv_W"] = beta2 * state["sv_W"] + (1 - beta2) * grad_W ** 2
    state["m_b"]  = beta1 * state["m_b"]  + (1 - beta1) * grad_b
    state["sv_b"] = beta2 * state["sv_b"] + (1 - beta2) * grad_b ** 2
    state["m_v"]  = beta1 * state["m_v"]  + (1 - beta1) * grad_v
    state["sv_v"] = beta2 * state["sv_v"] + (1 - beta2) * grad_v ** 2
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    W = W - learning_rate * (state["m_W"] / bc1) / (np.sqrt(state["sv_W"] / bc2) + eps)
    b = b - learning_rate * (state["m_b"] / bc1) / (np.sqrt(state["sv_b"] / bc2) + eps)
    v = v - learning_rate * (state["m_v"] / bc1) / (np.sqrt(state["sv_v"] / bc2) + eps)
    return W, b, v


def train_phase1(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    optimizer_name: str = "gd",
    learning_rate: float = 0.01,
    max_iterations: int = 1_000_000,
    loss_threshold: float,
    train_v: bool = True,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    report_every: Optional[int] = None,
) -> tuple:
    """
    Phase 1: train until loss ≤ loss_threshold, aborting if loss > threshold at t=10 000.

    If `report_every` is given, the network's overall min boundary distance is checkpointed
    at t=0 and every `report_every` steps, mirroring Phase 2's checkpointing so the two phases
    can be plotted on a single, absolute step axis.

    Returns
    -------
    W, b, v       : parameters at the threshold crossing (or at abort)
    t_threshold   : iteration when threshold was first reached (−1 if aborted/not reached)
    adam_state    : Adam momentum state dict to pass to train_phase2 (None for GD)
    stop_reason   : "threshold-reached", "loss-abort", or "max-iterations"
    checkpoints   : list of {"step": t, "loss": ..., "min_dist_large": ...} dicts
    """
    opt_name = optimizer_name.upper()
    if opt_name not in ("GD", "ADAM"):
        raise ValueError(f"Unknown optimizer {optimizer_name!r}. Use 'gd' or 'adam'.")

    adam_state: Optional[Dict[str, Any]] = None
    if opt_name == "ADAM":
        adam_state = {
            "t": 0,
            "m_W": np.zeros_like(W), "sv_W": np.zeros_like(W),
            "m_b": np.zeros_like(b), "sv_b": np.zeros_like(b),
            "m_v": np.zeros_like(v), "sv_v": np.zeros_like(v),
        }

    checkpoints: list = []

    for t in range(max_iterations + 1):
        preds = forward(W, b, v, X)
        loss_val = exp_loss(y, preds)

        if report_every and (t == 0 or t % report_every == 0):
            dists = compute_boundary_distances(W, b, v, X)
            fin = dists[np.isfinite(dists)]
            min_dist_large = float(np.min(fin)) if len(fin) > 0 else float("nan")
            checkpoints.append({"step": t, "loss": loss_val, "min_dist_large": min_dist_large})

        if t == 10_000 and loss_val > loss_threshold:
            print(f"[t={t}] Loss {loss_val:.6g} > threshold {loss_threshold:.6g} — aborting.")
            return W, b, v, -1, None, "loss-abort", checkpoints

        if loss_val <= loss_threshold and t > 1_000_000:
            print(f"[t={t}] Loss threshold reached: {loss_val:.6g}")
            return W, b, v, t, adam_state, "threshold-reached", checkpoints
            # dists = compute_boundary_distances(W, b, v, X)
            # dp = dists[y == 1]; dn = dists[y == -1]
            # fin_p = dp[np.isfinite(dp)]; fin_n = dn[np.isfinite(dn)]
            # if fin_p.size > 0 and fin_n.size > 0:
            #     dist_sum = float(np.min(fin_p)) + float(np.min(fin_n))
            #     print(f"dist_sum={dist_sum:.4g}")
            #     if 1.01 < dist_sum < 2.5:
            #         print(f"[t={t}] Loss threshold reached: {loss_val:.6g}, dist_sum={dist_sum:.4g}")
            #         return W, b, v, t, adam_state, "threshold-reached"

        if t == max_iterations:
            break

        grad_W, grad_b, grad_v = gradients(W, b, v, X, y)
        if not train_v:
            grad_v = np.zeros_like(grad_v)
        if opt_name == "GD":
            W = W - learning_rate * grad_W
            b = b - learning_rate * grad_b
            if train_v:
                v = v - learning_rate * grad_v
        else:
            W, b, v = _adam_step(W, b, v, grad_W, grad_b, grad_v, adam_state, learning_rate, beta1, beta2, eps)

        if t % 100_000 == 0 and t > 0:
            print(f"  [t={t}] loss={loss_val:.6g}")

    return W, b, v, -1, adam_state, "max-iterations", checkpoints
