"""
Debug visualizations for the 2-disk experiment.
"""

import matplotlib.pyplot as plt
import numpy as np

from .core import NetworkParams, network_forward
from .disk_network import forward as disk_forward


def plot_init_debug(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    small_params: NetworkParams,
    run_idx: int,
    t_threshold: int,
    output_path: str,
    x_lo: float = -3.0,
    x_hi: float = 3.0,
    title: str = "",
) -> None:
    """
    Overlay plot at the moment the small network is initialized.

    Large network: f_large(x₁, 0, …, 0) — value along the first coordinate
                   axis with all other coordinates set to zero.
    Small network: f_small(x) — the full 1D function.

    Both are plotted over [x_lo, x_hi] on the same axes.
    """
    x_range = np.linspace(x_lo, x_hi, 2000)
    d = W.shape[1]

    # ── Large network: probe along x₁ axis, all other coords = 0 ───────────
    X_probe = np.zeros((len(x_range), d))
    X_probe[:, 0] = x_range
    large_vals = disk_forward(W, b, v, X_probe)

    # ── Small network: continuous curve ─────────────────────────────────────
    small_vals = network_forward(small_params, x_range)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_range, large_vals,
        color="steelblue", linewidth=2.0, label=f"Large network  f(x₁, 0, …, 0)  [d={d}]",
    )
    ax.plot(
        x_range, small_vals,
        color="darkorange", linewidth=2.0, linestyle="--", label="Small network  f(x)  [1D]",
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline( 1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="x = +1 / −1")
    ax.axvline(-1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("x₁  (first coordinate / 1D network input)")
    ax.set_ylabel("Network output  f")
    ax.set_title(
        title or f"Init debug — run {run_idx},  t_threshold={t_threshold}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [debug] {output_path}")
