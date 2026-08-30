"""
2-disk classification experiment.

Flow
----
Phase 1 — large network only:
    Train until loss ≤ 1/n, or abort if loss > 1/n at t=10 000.

Phase 2 — both networks jointly for max_iterations steps:
    Initialize the 1D 2-neuron network from the large network's state at
    Phase 1 end (matching distance ratio and network values at closest
    training points), then train BOTH networks for max_iterations more
    steps.  Every report_every steps, compute the min distance to the
    decision boundary for each network.  At the end, plot both curves on
    the same graph.

Building blocks
---------------
disk_dataset.py   — dataset sampling
disk_network.py   — forward pass, loss, gradients, boundary distance
disk_trainer.py   — train_phase1
core.py           — NetworkParams, gradient_descent_step (for 1D network)
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from .core import (
    NetworkParams,
    exponential_loss,
    gradient_descent_step,
    network_forward,
)
from .disk_dataset import create_disk_dataset
from .disk_debug import plot_init_debug
from .disk_network import (
    compute_boundary_distances,
    exp_loss,
    forward as disk_forward,
    gradients,
)
from .disk_trainer import train_phase1


# ---------------------------------------------------------------------------
# 1D boundary-distance helper
# ---------------------------------------------------------------------------

def _compute_1d_boundary_distances(params: NetworkParams, x_data: np.ndarray) -> np.ndarray:
    """
    Return the distance from each point in x_data to the nearest zero of the
    1D piecewise-linear network  f(x) = sum_j v_j * ReLU(w_j*x + b_j).

    Uses exact piecewise-linear analysis (no numerical grid search).
    """
    w, b, v = params.w, params.b, params.v
    x_data = np.asarray(x_data, dtype=float)

    # Breakpoints where each neuron's pre-activation crosses zero.
    bps = []
    for j in range(len(w)):
        if abs(w[j]) > 1e-15:
            bps.append(-b[j] / w[j])
    bps = sorted(set(bps))
    region_limits = [-np.inf] + bps + [np.inf]

    zeros: list[float] = []

    for i in range(len(region_limits) - 1):
        lo, hi = region_limits[i], region_limits[i + 1]

        # A representative interior point for this region.
        if np.isinf(lo):
            x_test = hi - 1.0
        elif np.isinf(hi):
            x_test = lo + 1.0
        else:
            x_test = (lo + hi) / 2.0

        act = (w * x_test + b > 0).astype(float)
        A = float(np.dot(v * act, w))
        C = float(np.dot(v * act, b))

        if abs(A) > 1e-15:
            x_zero = -C / A
            lo_ok = np.isinf(lo) or x_zero >= lo - 1e-12
            hi_ok = np.isinf(hi) or x_zero <= hi + 1e-12
            if lo_ok and hi_ok:
                zeros.append(x_zero)
        elif abs(C) < 1e-10:
            # f = 0 throughout this region; any training point inside has dist 0.
            for x_tp in x_data:
                in_lo = np.isinf(lo) or x_tp >= lo
                in_hi = np.isinf(hi) or x_tp <= hi
                if in_lo and in_hi:
                    zeros.append(float(x_tp))

    if not zeros:
        return np.full(len(x_data), np.inf)

    z = np.array(zeros)
    return np.array([float(np.min(np.abs(x - z))) for x in x_data])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment_disks(
    n: int = 100,
    d: int = 10,
    k: int = 4,
    optimizer_name: str = "gd",
    learning_rate: float = 0.01,
    max_iterations: int = 1_000_000,
    report_every: int = 10_000,
    seed: int = 42,
    num_runs: int = 1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    output_csv: str = "experiment_disks_summary.csv",
    dist_comparison_csv: str = "experiment_disks_dist_comparison.csv",
    dist_comparison_png: str = "",
    weight_diff_csv: str = "experiment_disks_weight_diff.csv",
    unified_comparison_csv: str = "experiment_disks_unified_comparison.csv",
):
    """
    Run the 2-disk experiment and produce a boundary-distance comparison plot.

    output_csv               — checkpoint rows: run, step, t, loss, min_dist_large,
                               min_dist_pos, min_dist_neg, min_dist_small
    dist_comparison_csv      — Phase-2-relative (run, step, min_dist_large, min_dist_small);
                               step resets to 0 at the start of Phase 2.
    dist_comparison_png      — distance-vs-step line plot for both networks
    unified_comparison_csv   — (run, step, min_dist_large, min_dist_small) on a single
                               absolute step axis spanning Phase 1 and Phase 2. min_dist_small
                               is blank for Phase-1 rows, since the small network does not
                               exist yet; it starts appearing once Phase 2 begins (step ≈
                               t_threshold, the iteration at which Phase 1's loss threshold
                               was reached).
    """
    loss_threshold = 1.0 / n
    opt_name = optimizer_name.upper()
    if not dist_comparison_png:
        dist_comparison_png = (
            f"experiment_disks_dist_comparison_{opt_name.lower()}_k{k}_"
            f"lr{learning_rate}_n{n}_d{d}_runs{num_runs}.png"
        )

    print(
        f"experiment_disks: n={n}, d={d}, k={k}, "
        f"opt={opt_name}, lr={learning_rate}, "
        f"num_runs={num_runs}, seed={seed}"
    )
    print(f"Loss threshold (1/n): {loss_threshold:.6g}")


    all_summary_rows: list[dict] = []
    all_comparison_rows: list[dict] = []
    all_weight_diff_rows: list[dict] = []
    all_unified_rows: list[dict] = []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        print(f"\n--- Run {run_idx} (seed={run_seed}) ---")

        X, y = create_disk_dataset(n, d, seed=run_seed)

        rng = np.random.default_rng(run_seed + 1)
        W = rng.standard_normal((k, d)) * np.sqrt(2.0 / d)
        b = np.zeros(k)
        v = np.concatenate([np.ones(k // 2), -np.ones(k - k // 2)])

        # ── Phase 1: large network → loss threshold ─────────────────────────
        W, b, v, t_threshold, adam_state, phase1_reason, phase1_checkpoints = train_phase1(
            W, b, v, X, y,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            loss_threshold=loss_threshold,
            train_v=False,
            beta1=beta1, beta2=beta2, eps=eps,
            report_every=report_every,
        )

        if phase1_reason == "loss-abort":
            print(f"  Run {run_idx}: loss-abort — skipping.")
            continue

        # ── Initialize 1D network from large network's state at threshold ───
        dists = compute_boundary_distances(W, b, v, X)
        dists_pos = dists[y == 1]
        dists_neg = dists[y == -1]
        fin_pos = np.isfinite(dists_pos)
        fin_neg = np.isfinite(dists_neg)

        if not fin_pos.any() or not fin_neg.any():
            print(f"  Run {run_idx}: no finite boundary distances — skipping.")
            continue

        # Phase-1 checkpoints, on the same absolute step axis as Phase 2 below.
        # min_dist_small is left blank: the small network is not initialized until
        # Phase 1 ends (see below), so it has no margin to report during Phase 1.
        for cp in phase1_checkpoints:
            all_unified_rows.append({
                "run": run_idx, "step": cp["step"],
                "min_dist_large": cp["min_dist_large"],
                "min_dist_small": "",
            })

        X_pos = X[y == 1]
        X_neg = X[y == -1]
        argmin_pos = int(np.argmin(dists_pos[fin_pos]))
        argmin_neg = int(np.argmin(dists_neg[fin_neg]))
        min_dp = float(dists_pos[fin_pos][argmin_pos])
        min_dn = float(dists_neg[fin_neg][argmin_neg])
        net_val_pos = float(disk_forward(W, b, v, X_pos[fin_pos][argmin_pos].reshape(1, -1))[0])
        net_val_neg = float(disk_forward(W, b, v, X_neg[fin_neg][argmin_neg].reshape(1, -1))[0])

        # x_0: zero-crossing s.t. min_dist_small == min_dist_large.
        # Condition w1+w2+b1=b2 (with shared breakpoints) forces f(1)=-f(-1);
        # scale is set from the value at the closer-class training point.
        if min_dp <= min_dn:
            x_0 = 1.0 - min_dp
            w1_0 = net_val_pos / min_dp
            w2_0 = -net_val_pos / (2.0 - min_dp)
        else:
            x_0 = min_dn - 1.0
            w2_0 = net_val_neg / min_dn
            w1_0 = -net_val_neg / (2.0 - min_dn)
        b1_0 = -w1_0 * x_0
        b2_0 = -w2_0 * x_0

        x_1d = np.array([1.0, -1.0])
        y_1d = np.array([1.0, -1.0])

        small_params = NetworkParams(
            w=np.array([w1_0, w2_0]),
            b=np.array([b1_0, b2_0]),
            v=np.array([1.0, -1.0]),
        )
        small_opt_state = (
            {"beta1": beta1, "beta2": beta2, "eps": eps} if opt_name == "ADAM" else None
        )

        print(
            f"  [1D init] min_dp={min_dp:.4g}  min_dn={min_dn:.4g}  x0={x_0:.4g}  "
            f"w=[{w1_0:.4g}, {w2_0:.4g}]  b=[{b1_0:.4g}, {b2_0:.4g}]"
        )

        # plot_init_debug(
        #     W, b, v, small_params,
        #     run_idx=run_idx,
        #     t_threshold=t_threshold,
        #     output_path=f"experiment_disks_init_debug_d{d}_k{k}_run{run_idx}.png",
        # )

        # ── Phase 2: joint training for max_iterations steps ────────────────
        prev_W: np.ndarray | None = None
        prev_b: np.ndarray | None = None
        prev_v: np.ndarray | None = None

        for step in range(max_iterations + 1):
            t_large = t_threshold + step

            # Checkpoint: at step 0 and every report_every steps.
            if step == 0 or step % report_every == 0:
                # Large network distances
                cdists = compute_boundary_distances(W, b, v, X)
                fin_all = cdists[np.isfinite(cdists)]
                min_dist_large = float(np.min(fin_all)) if len(fin_all) > 0 else float("nan")
                cdp = cdists[y == 1]; fin_p = cdp[np.isfinite(cdp)]
                cdn = cdists[y == -1]; fin_n = cdn[np.isfinite(cdn)]
                min_dist_pos = float(np.min(fin_p)) if len(fin_p) > 0 else float("nan")
                min_dist_neg = float(np.min(fin_n)) if len(fin_n) > 0 else float("nan")

                # Large network loss
                loss_large = float(exp_loss(y, disk_forward(W, b, v, X)))

                # Small network distances and loss
                sd = _compute_1d_boundary_distances(small_params, x_1d)
                fin_sd = sd[np.isfinite(sd)]
                min_dist_small = float(np.min(fin_sd)) if len(fin_sd) > 0 else float("nan")
                loss_small = float(exponential_loss(y_1d, network_forward(small_params, x_1d)))

                print(
                    f"  [run={run_idx} step={step}] "
                    f"loss_L={loss_large:.4g} dist_L={min_dist_large:.4g}  "
                    f"loss_S={loss_small:.4g} dist_S={min_dist_small:.4g}"
                )

                all_summary_rows.append({
                    "run": run_idx, "step": step,
                    "loss_large": loss_large, "loss_small": loss_small,
                    "min_dist_large": min_dist_large,
                    "min_dist_pos": min_dist_pos, "min_dist_neg": min_dist_neg,
                    "min_dist_small": min_dist_small,
                })
                all_comparison_rows.append({
                    "run": run_idx, "step": step,
                    "min_dist_large": min_dist_large,
                    "min_dist_small": min_dist_small,
                })
                all_unified_rows.append({
                    "run": run_idx, "step": t_large,
                    "min_dist_large": min_dist_large,
                    "min_dist_small": min_dist_small,
                })

                if prev_W is not None:
                    all_weight_diff_rows.append({
                        "run": run_idx, "step": step,
                        "delta_W": W - prev_W,
                        "delta_b": b - prev_b,
                        "delta_v": v - prev_v,
                    })
            prev_W = W.copy()
            prev_b = b.copy()
            prev_v = v.copy()

            if step == max_iterations:
                break

            # ── Large network gradient step ──────────────────────────────────
            grad_W, grad_b, grad_v = gradients(W, b, v, X, y)
            if opt_name == "GD":
                W = W - learning_rate * grad_W
                b = b - learning_rate * grad_b
            else:  # ADAM — continue from Phase 1 state
                adam_state["t"] += 1
                t_a = adam_state["t"]
                adam_state["m_W"]  = beta1 * adam_state["m_W"]  + (1 - beta1) * grad_W
                adam_state["sv_W"] = beta2 * adam_state["sv_W"] + (1 - beta2) * grad_W ** 2
                adam_state["m_b"]  = beta1 * adam_state["m_b"]  + (1 - beta1) * grad_b
                adam_state["sv_b"] = beta2 * adam_state["sv_b"] + (1 - beta2) * grad_b ** 2
                bc1 = 1.0 - beta1 ** t_a
                bc2 = 1.0 - beta2 ** t_a
                W = W - learning_rate * (adam_state["m_W"] / bc1) / (np.sqrt(adam_state["sv_W"] / bc2) + eps)
                b = b - learning_rate * (adam_state["m_b"] / bc1) / (np.sqrt(adam_state["sv_b"] / bc2) + eps)

            # ── Small network gradient step ──────────────────────────────────
            small_params, _, _ = gradient_descent_step(
                small_params, x_1d, y_1d, learning_rate,
                optimizer_name=optimizer_name,
                optimizer_state=small_opt_state,
            )
            small_params.v = np.array([1.0, -1.0])  # keep v=[1,-1] fixed

        t_final = t_threshold + max_iterations
        # plot_init_debug(
        #     W, b, v, small_params,
        #     run_idx=run_idx,
        #     t_threshold=t_final,
        #     output_path=f"experiment_disks_final_debug_d{d}_k{k}_run{run_idx}.png",
        #     title=f"End of training — run {run_idx},  t={t_final}",
        # )

        print(f"  Run {run_idx}: Phase 2 done ({max_iterations} steps).")

    # ── Write CSVs ──────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "step", "t",
            "loss_large", "loss_small",
            "min_dist_large", "min_dist_pos", "min_dist_neg", "min_dist_small",
        ])
        writer.writeheader()
        writer.writerows(all_summary_rows)

    with open(dist_comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "step", "min_dist_large", "min_dist_small"])
        writer.writeheader()
        writer.writerows(all_comparison_rows)

    with open(weight_diff_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "step", "delta_W", "delta_b", "delta_v"])
        writer.writeheader()
        writer.writerows(all_weight_diff_rows)

    with open(unified_comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "step", "min_dist_large", "min_dist_small"])
        writer.writeheader()
        writer.writerows(all_unified_rows)

    # ── Plot: distance vs step ───────────────────────────────────────────────
    if all_comparison_rows:
        from collections import defaultdict

        num_completed = len({r["run"] for r in all_comparison_rows})
        fig, ax = plt.subplots(figsize=(10, 6))

        if num_completed <= 5:
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
            by_run: dict = defaultdict(lambda: {"steps": [], "large": [], "small": []})
            for row in all_comparison_rows:
                by_run[row["run"]]["steps"].append(row["step"])
                by_run[row["run"]]["large"].append(row["min_dist_large"])
                by_run[row["run"]]["small"].append(row["min_dist_small"])

            for i, run_id in enumerate(sorted(by_run)):
                ls = linestyles[i % len(linestyles)]
                run_label = f" (run {run_id})" if num_completed > 1 else ""
                ax.plot(by_run[run_id]["steps"], by_run[run_id]["large"],
                        color="steelblue", linestyle=ls,
                        label=f"Large network ({d}D, k={k}){run_label}")
                ax.plot(by_run[run_id]["steps"], by_run[run_id]["small"],
                        color="tomato", linestyle=ls,
                        label=f"Small network (1D, k=2){run_label}")
        else:
            by_step: dict = defaultdict(lambda: {"large": [], "small": []})
            for row in all_comparison_rows:
                by_step[row["step"]]["large"].append(row["min_dist_large"])
                by_step[row["step"]]["small"].append(row["min_dist_small"])

            steps_sorted = sorted(by_step)
            mean_large = np.array([float(np.nanmean(by_step[s]["large"])) for s in steps_sorted])
            mean_small = np.array([float(np.nanmean(by_step[s]["small"])) for s in steps_sorted])
            std_large  = np.array([float(np.nanstd(by_step[s]["large"]))  for s in steps_sorted])
            std_small  = np.array([float(np.nanstd(by_step[s]["small"]))  for s in steps_sorted])

            ax.plot(steps_sorted, mean_large, label=f"Large network ({d}D, k={k})", color="steelblue")
            ax.plot(steps_sorted, mean_small, label="Small network (1D, k=2)", color="tomato")
            half_std_large = 0.5 * std_large
            half_std_small = 0.5 * std_small
            ax.fill_between(steps_sorted, mean_large - half_std_large, mean_large + half_std_large,
                            color="steelblue", alpha=0.2)
            ax.fill_between(steps_sorted, mean_small - half_std_small, mean_small + half_std_small,
                            color="tomato", alpha=0.2)

        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("Step (Phase 2)")
        ax.set_ylabel("Min boundary distance")
        ax.set_title(
            f"Min boundary distance vs steps  —  "
            f"{opt_name}, lr={learning_rate}, n={n}, d={d}, k={k}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(dist_comparison_png, dpi=200)
        plt.close(fig)
        print(f"  {dist_comparison_png}")

    non_aborted = len({r["run"] for r in all_comparison_rows})
    print(f"\nDone. {non_aborted}/{num_runs} non-aborted runs.")
    print(f"  {output_csv}")
    print(f"  {dist_comparison_csv}")
    print(f"  {unified_comparison_csv}")

    return all_summary_rows
