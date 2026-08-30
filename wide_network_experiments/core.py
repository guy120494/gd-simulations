import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass
class NetworkParams:
    """Parameters for a 1D ReLU network with k neurons: Phi(x)=sum_j v_j ReLU(w_j x + b_j)."""

    w: np.ndarray  # shape (k,)
    b: np.ndarray  # shape (k,)
    v: np.ndarray  # shape (k,)

    def __post_init__(self):
        assert len(self.w) == len(self.b) == len(self.v)

    @property
    def k(self) -> int:
        return len(self.w)

    def copy(self) -> "NetworkParams":
        return NetworkParams(w=self.w.copy(), b=self.b.copy(), v=self.v.copy())


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def network_forward(params: NetworkParams, x: np.ndarray) -> np.ndarray:
    """
    Forward pass for 1D inputs.

    Args:
        params: network parameters
        x: array-like of shape (n,)

    Returns:
        outputs of shape (n,)
    """
    x = np.asarray(x).reshape(-1)
    # (k, n) pre-activations
    pre = params.w[:, None] * x[None, :] + params.b[:, None]
    act = relu(pre)
    return (params.v[:, None] * act).sum(axis=0)


def exponential_loss(y: np.ndarray, predictions: np.ndarray) -> float:
    """Mean exponential loss: mean_i exp(-y_i * f(x_i))."""
    return float(np.mean(np.exp(-y * predictions)))


def compute_gradients(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable = exponential_loss,
) -> NetworkParams:
    """
    Compute gradients of mean exponential loss wrt (w,b,v).

    Note: `loss_fn` is currently unused (kept for API compatibility).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    k = params.k
    n = len(x)

    pre = params.w[:, None] * x[None, :] + params.b[:, None]  # (k, n)
    relu_pre = relu(pre)  # (k, n)
    outputs = (params.v[:, None] * relu_pre).sum(axis=0)  # (n,)

    # dL/dout_i = -(1/n) y_i exp(-y_i out_i)
    dL_dout = -(1.0 / n) * y * np.exp(-y * outputs)  # (n,)

    indicator = (pre > 0).astype(float)  # (k, n)

    grad_v = (dL_dout[None, :] * relu_pre).sum(axis=1)  # (k,)
    grad_w = (dL_dout[None, :] * params.v[:, None] * indicator * x[None, :]).sum(axis=1)  # (k,)
    grad_b = (dL_dout[None, :] * params.v[:, None] * indicator).sum(axis=1)  # (k,)

    return NetworkParams(w=grad_w, b=grad_b, v=grad_v)


def gradient_descent_step(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    optimizer_name: str = "GD",
    optimizer_state: Optional[Dict[str, object]] = None,
    loss_fn: Callable = exponential_loss,
) -> Tuple[NetworkParams, float, Optional[Dict[str, np.ndarray]]]:
    grads = compute_gradients(params, x, y, loss_fn)

    opt_name = optimizer_name.upper()
    adam_info: Optional[Dict[str, np.ndarray]] = None
    if opt_name == "GD":
        new_params = NetworkParams(
            w=params.w - learning_rate * grads.w,
            b=params.b - learning_rate * grads.b,
            v=params.v if params.k == 2 else params.v - learning_rate * grads.v,
        )
    elif opt_name == "ADAM":
        # NumPy implementation of ADAM update.
        if optimizer_state is None:
            optimizer_state = {}
        beta1 = float(optimizer_state.get("beta1", 0.9))
        beta2 = float(optimizer_state.get("beta2", 0.999))
        eps = float(optimizer_state.get("eps", 1e-8))
        t = int(optimizer_state.get("t", 0)) + 1
        optimizer_state["t"] = t

        if "m_w" not in optimizer_state:
            optimizer_state["m_w"] = np.zeros_like(params.w)
            optimizer_state["v_w"] = np.zeros_like(params.w)
            optimizer_state["m_b"] = np.zeros_like(params.b)
            optimizer_state["v_b"] = np.zeros_like(params.b)
            optimizer_state["m_v"] = np.zeros_like(params.v)
            optimizer_state["v_v"] = np.zeros_like(params.v)

        optimizer_state["m_w"] = beta1 * optimizer_state["m_w"] + (1.0 - beta1) * grads.w
        optimizer_state["v_w"] = beta2 * optimizer_state["v_w"] + (1.0 - beta2) * (grads.w * grads.w)
        optimizer_state["m_b"] = beta1 * optimizer_state["m_b"] + (1.0 - beta1) * grads.b
        optimizer_state["v_b"] = beta2 * optimizer_state["v_b"] + (1.0 - beta2) * (grads.b * grads.b)
        optimizer_state["m_v"] = beta1 * optimizer_state["m_v"] + (1.0 - beta1) * grads.v
        optimizer_state["v_v"] = beta2 * optimizer_state["v_v"] + (1.0 - beta2) * (grads.v * grads.v)

        m_w_hat = optimizer_state["m_w"] / (1.0 - beta1**t)
        v_w_hat = optimizer_state["v_w"] / (1.0 - beta2**t)
        m_b_hat = optimizer_state["m_b"] / (1.0 - beta1**t)
        v_b_hat = optimizer_state["v_b"] / (1.0 - beta2**t)
        m_v_hat = optimizer_state["m_v"] / (1.0 - beta1**t)
        v_v_hat = optimizer_state["v_v"] / (1.0 - beta2**t)

        new_w = params.w - learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
        new_b = params.b - learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
        new_v = params.v if params.k == 2 else params.v - learning_rate * m_v_hat / (np.sqrt(v_v_hat) + eps)
        new_params = NetworkParams(w=new_w, b=new_b, v=new_v)
        adam_info = {
            "v_w_hat": v_w_hat,
            "v_b_hat": v_b_hat,
        }
    else:
        raise ValueError(f"Unsupported optimizer_name={optimizer_name}. Use 'GD' or 'ADAM'.")

    loss = loss_fn(np.asarray(y), network_forward(new_params, np.asarray(x)))
    return new_params, loss, adam_info

