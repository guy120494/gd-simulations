import multiprocessing
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt

import csv
from collections import deque
from collections import defaultdict
from pathlib import Path

from .core import NetworkParams, network_forward, exponential_loss, gradient_descent_step
from .datasets import create_dataset
from .init_utils import initialize_network


def _format_lr_tag(learning_rate: float) -> str:
    """Return a filesystem-safe tag for learning rates."""
    return f"{learning_rate:.6g}".replace("-", "m").replace(".", "p")


def _load_avg_curve(avg_csv_path: str):
    path = Path(avg_csv_path)
    if not path.exists():
        return None

    t_vals = []
    mean_vals = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_vals.append(int(row["t"]))
            mean_vals.append(float(row["mean_w_diff"]))
    return np.asarray(t_vals, dtype=int), np.asarray(mean_vals, dtype=float)


def experiment_5f_hit_linear_condition_with_low_loss(
    num_runs: int = 10000,
    max_iterations: int = 10_000_000,
    learning_rate: float = 0.05,
    optimizer_name: str = "gd",
    tol: float = 1e-3,
    loss_threshold: float = 0.5,
    seed: int = 42,
    beta1: float = 0.9,
    beta2: float = 0.999,
    sample_every: int = 1000,
    track_weight_diff: bool = True,
    track_convergence: bool = True,
    output_dir: str = ".",
):
    """
    Runs num_runs times with random init:
        w ~ N(0,2), b=0, k=2, v=[1,-1] fixed.

    Stop when:
        GD: |w1 + b1 + w2 - b2| < tol AND loss < loss_threshold
        Adam: std_{t-999..t}(b2 - b1) < 0.1

    Abort rule:
        If at t=10000 loss still not < loss_threshold → abort.

    Writes:
        experiment_5f_runs.csv
        experiment_5f_summary.txt
        experiment_5f_hit_time_hist.png
        experiment_5f_metric_hist.png
        experiment_5f_hit_time_hist.csv
        experiment_5f_metric_hist.csv
        experiment_5f_convergence_delta_hist_{opt}.png/csv  (if track_convergence)
        experiment_5f_convergence_std_hist_{opt}.png/csv    (if track_convergence)
    """

    rng = np.random.default_rng(seed)
    x, y = create_dataset(symmetric=True)

    if sample_every <= 0:
        raise ValueError("sample_every must be a positive integer.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hit_times = np.full(num_runs, -1, dtype=int)
    metric_values = []
    trajectory_records = []
    adam_bias_abs_values = []

    count_hit = 0
    count_loss_abort = 0
    count_max_iterations = 0

    convergence_deltas = []
    convergence_stds = []

    opt_name = optimizer_name.upper()
    lr_tag = _format_lr_tag(learning_rate)

    runs_csv_path = out / "experiment_5f_runs.csv"
    summary_txt_path = out / "experiment_5f_summary.txt"
    hist1_csv_path = out / "experiment_5f_hit_time_hist.csv"
    hist2_csv_path = out / "experiment_5f_metric_hist.csv"
    trajectory_csv_path = out / f"experiment_5f_wdiff_trajectory_{opt_name.lower()}_lr{lr_tag}.csv"
    wdiff_avg_csv_path = out / f"experiment_5f_wdiff_avg_{opt_name.lower()}_lr{lr_tag}.csv"
    wdiff_avg_png_path = out / f"experiment_5f_wdiff_avg_{opt_name.lower()}_lr{lr_tag}.png"
    compare_png_path = out / f"experiment_5f_wdiff_avg_compare_lr{lr_tag}.png"
    adam_bias_hist_csv_path = out / f"experiment_5f_bias_abs_hist_adam_lr{lr_tag}.csv"
    adam_bias_hist_png_path = out / f"experiment_5f_bias_abs_hist_adam_lr{lr_tag}.png"

    with open(runs_csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "run", "stop_reason", "t_last", "t_hit",
                "w1_0", "b1_0", "w2_0", "b2_0",
                "w1_T", "b1_T", "w2_T", "b2_T",
                "loss_last", "expr_last",
                "metric_min",
                "adam_min_norm_vhat",
                "final_bias_abs",
                "convergence_delta",
                "convergence_std",
            ],
        )
        writer.writeheader()

        for r in range(num_runs):
            print(f"Run: {r}")

            # Initialization
            w1_0 = float(rng.normal(0.0, np.sqrt(2)))
            b1_0 = 0.0
            w2_0 = float(rng.normal(0.0, np.sqrt(2)))
            b2_0 = 0.0

            params = initialize_network(
                k=2,
                init_type="thesis",
                seed=None,
                w1_init=w1_0,
                b1_init=b1_0,
                w2_init=w2_0,
                b2_init=b2_0,
            )

            params.v = np.array([1.0, -1.0], dtype=float)

            t = 0
            stop_reason = None
            optimizer_state = {"beta1": beta1, "beta2": beta2} if opt_name == "ADAM" else None

            diff_history = deque(maxlen=1000) if opt_name == "ADAM" else None
            min_vhat_norm = float("inf")
            run_samples = []
            sampled_iterations = set()
            # Tracks last 11 samples of |b2-b1| → covers last 10*sample_every iterations
            conv_deque = deque(maxlen=11) if track_convergence else None

            while t <= max_iterations:

                preds = network_forward(params, x)
                loss_t = float(exponential_loss(y, preds))

                w1 = float(params.w[0])
                b1 = float(params.b[0])
                w2 = float(params.w[1])
                b2 = float(params.b[1])

                expr_t = abs(w1 + b1 + w2 - b2)
                w_diff_t = w1 - w2
                bias_diff_abs_t = abs(b2 - b1)

                if t % sample_every == 0:
                    if track_weight_diff:
                        run_samples.append((t, w1, w2, b1, b2, loss_t))
                        sampled_iterations.add(t)
                    if track_convergence:
                        conv_deque.append(bias_diff_abs_t)

                # Abort rule
                if t == 10_000 and not (loss_t < loss_threshold):
                    stop_reason = "loss-abort"
                    count_loss_abort += 1
                    break

                # Hit condition
                if opt_name == "GD":
                    if loss_t < loss_threshold and expr_t < tol:
                        hit_times[r] = t
                        stop_reason = "hit condition"
                        count_hit += 1

                        metric_min = min(abs(b2 - b1), abs(w1_0 + w2_0) / 2.0)
                        metric_values.append(metric_min)
                        break
                elif opt_name == "ADAM":
                    # Track (b2 - b1) over the last 1000 iterations and check convergence.
                    assert diff_history is not None
                    diff_history.append(abs(b2 - b1))

                    if len(diff_history) == diff_history.maxlen:
                        std_diff = float(np.std(np.asarray(diff_history)))
                        if std_diff < 1e-7:
                            hit_times[r] = t
                            stop_reason = "hit condition"
                            count_hit += 1

                            metric_values.append(abs(b2 - b1))
                            break
                else:
                    raise ValueError(f"Unsupported optimizer_name={optimizer_name}. Use 'gd' or 'adam'.")

                if t == max_iterations:
                    stop_reason = "hit condition"
                    count_max_iterations += 1
                    break

                params, _, adam_info = gradient_descent_step(
                    params,
                    x,
                    y,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    optimizer_state=optimizer_state,
                )
                params.v = np.array([1.0, -1.0], dtype=float)

                # For Adam runs, track the minimal ||[v_w_hat, v_b_hat]|| along the run.
                if opt_name == "ADAM" and adam_info is not None:
                    v_w_hat = adam_info["v_w_hat"]
                    v_b_hat = adam_info["v_b_hat"]
                    cur_norm = float(np.linalg.norm(np.concatenate([v_w_hat, v_b_hat])))
                    min_vhat_norm = min(min_vhat_norm, cur_norm)

                t += 1

            # Final parameters
            w1_T = float(params.w[0])
            b1_T = float(params.b[0])
            w2_T = float(params.w[1])
            b2_T = float(params.b[1])
            final_bias_abs = abs(b2_T - b1_T)
            if opt_name == "ADAM":
                adam_bias_abs_values.append(final_bias_abs)

            # Convergence check: only meaningful for runs that reached max_iterations
            if track_convergence and stop_reason != "loss-abort" and conv_deque is not None and len(conv_deque) == conv_deque.maxlen:
                arr = np.array(list(conv_deque))
                convergence_deltas.append(float(abs(arr[-1] - arr[0])))
                convergence_stds.append(float(np.std(arr)))

            if track_weight_diff and (t not in sampled_iterations):
                run_samples.append((t, w1_T, w2_T, b1_T, b2_T, loss_t))

            metric_val = ""
            if stop_reason == "hit condition":
                # metric_val = metric_values[-1]
                metric_val = "NOT IMPORTANT"

            adam_min_norm_val = ""
            if opt_name == "ADAM" and min_vhat_norm != float("inf"):
                adam_min_norm_val = float(min_vhat_norm)

            conv_delta_val = ""
            conv_std_val = ""
            if track_convergence and stop_reason != "loss-abort" and conv_deque is not None and len(conv_deque) == conv_deque.maxlen:
                arr = np.array(list(conv_deque))
                conv_delta_val = float(abs(arr[-1] - arr[0]))
                conv_std_val = float(np.std(arr))

            writer.writerow({
                "run": r,
                "stop_reason": stop_reason,
                "t_last": t,
                "t_hit": hit_times[r],
                "w1_0": w1_0,
                "b1_0": b1_0,
                "w2_0": w2_0,
                "b2_0": b2_0,
                "w1_T": w1_T,
                "b1_T": b1_T,
                "w2_T": w2_T,
                "b2_T": b2_T,
                "loss_last": loss_t,
                "expr_last": expr_t,
                "metric_min": metric_val,
                "adam_min_norm_vhat": adam_min_norm_val,
                "final_bias_abs": final_bias_abs,
                "convergence_delta": conv_delta_val,
                "convergence_std": conv_std_val,
            })
            if track_weight_diff:
                for t_sample, w_1_sample, w_2_sample, b_1_sample, b_2_sample, loss_sample in run_samples:
                    trajectory_records.append({
                        "run": r,
                        "optimizer": opt_name,
                        "learning_rate": learning_rate,
                        "t": t_sample,
                        "w_1": w_1_sample,
                        "w_2": w_2_sample,
                        "b_1": b_1_sample,
                        "b_2": b_2_sample,
                        "loss": loss_sample,
                        "stop_reason": stop_reason,
                    })

    # Safety check
    assert count_hit + count_loss_abort + count_max_iterations == num_runs

    # -------------------------
    # Histogram 1: Hit times
    # -------------------------
    successful_hits = hit_times[hit_times >= 0]

    hist1_counts = None
    hist1_edges = None
    if len(successful_hits) > 0:
        hist1_counts, hist1_edges = np.histogram(successful_hits, bins=40)

        plt.figure(figsize=(8, 5))
        plt.hist(successful_hits, bins=40)
        plt.title("Hit time histogram")
        plt.xlabel("Iterations")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / "experiment_5f_hit_time_hist.png", dpi=200)
        plt.close()

        with open(hist1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist1_counts)):
                writer.writerow([
                    hist1_edges[i],
                    hist1_edges[i+1],
                    hist1_counts[i]
                ])

    # -------------------------
    # Histogram 2: Metric
    # -------------------------
    metric_arr = np.array(metric_values)

    hist2_counts = None
    hist2_edges = None
    if len(metric_arr) > 0:
        hist2_counts, hist2_edges = np.histogram(metric_arr, bins=40)

        plt.figure(figsize=(8, 5))
        plt.hist(metric_arr, bins=40)
        plt.title("Histogram of min(|b2-b1|, |w1^0+w2^0|/2)")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / "experiment_5f_metric_hist.png", dpi=200)
        plt.close()

        with open(hist2_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist2_counts)):
                writer.writerow([
                    hist2_edges[i],
                    hist2_edges[i+1],
                    hist2_counts[i]
                ])

    # -------------------------
    # Histogram 3: Convergence delta (Option A)
    # -------------------------
    if track_convergence and len(convergence_deltas) > 0:
        delta_arr = np.array(convergence_deltas)
        delta_counts, delta_edges = np.histogram(delta_arr, bins=40)
        conv_delta_csv = out / f"experiment_5f_convergence_delta_hist_{opt_name.lower()}_lr{lr_tag}.csv"
        conv_delta_png = out / f"experiment_5f_convergence_delta_hist_{opt_name.lower()}_lr{lr_tag}.png"

        plt.figure(figsize=(8, 5))
        plt.hist(delta_arr, bins=40)
        plt.title(f"Option A: |b2-b1| change over last {10 * sample_every} iters ({opt_name}, lr={learning_rate})")
        plt.xlabel("|Δ|b2-b1||")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(conv_delta_png, dpi=200)
        plt.close()

        with open(conv_delta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(delta_counts)):
                writer.writerow([delta_edges[i], delta_edges[i + 1], delta_counts[i]])

    # -------------------------
    # Histogram 4: Convergence std (Option B)
    # -------------------------
    if track_convergence and len(convergence_stds) > 0:
        std_arr = np.array(convergence_stds)
        std_counts, std_edges = np.histogram(std_arr, bins=40)
        conv_std_csv = out / f"experiment_5f_convergence_std_hist_{opt_name.lower()}_lr{lr_tag}.csv"
        conv_std_png = out / f"experiment_5f_convergence_std_hist_{opt_name.lower()}_lr{lr_tag}.png"

        plt.figure(figsize=(8, 5))
        plt.hist(std_arr, bins=40)
        plt.title(f"Option B: std(|b2-b1|) over last {10 * sample_every} iters ({opt_name}, lr={learning_rate})")
        plt.xlabel("std(|b2-b1|)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(conv_std_png, dpi=200)
        plt.close()

        with open(conv_std_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(std_counts)):
                writer.writerow([std_edges[i], std_edges[i + 1], std_counts[i]])

    avg_counts = None
    avg_t_values = None
    avg_mean_wdiff = None
    if track_weight_diff and len(trajectory_records) > 0:
        with open(trajectory_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run",
                    "optimizer",
                    "learning_rate",
                    "t",
                    "w_1",
                    "w_2",
                    "b_1",
                    "b_2",
                    "loss",
                    "stop_reason",
                ],
            )
            writer.writeheader()
            writer.writerows(trajectory_records)

        grouped_wdiff = defaultdict(list)
        for row in trajectory_records:
            w_diff = float(row["w_1"]) - float(row["w_2"])
            grouped_wdiff[int(row["t"])].append(w_diff)

        avg_t_values = np.array(sorted(grouped_wdiff.keys()), dtype=int)
        avg_mean_wdiff = np.array(
            [float(np.mean(grouped_wdiff[t])) for t in avg_t_values],
            dtype=float,
        )
        avg_std_wdiff = np.array(
            [float(np.std(grouped_wdiff[t])) for t in avg_t_values],
            dtype=float,
        )
        avg_counts = np.array([len(grouped_wdiff[t]) for t in avg_t_values], dtype=int)

        with open(wdiff_avg_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "mean_w_diff", "std_w_diff", "count"])
            for idx in range(len(avg_t_values)):
                writer.writerow([
                    int(avg_t_values[idx]),
                    float(avg_mean_wdiff[idx]),
                    float(avg_std_wdiff[idx]),
                    int(avg_counts[idx]),
                ])

        plt.figure(figsize=(8, 5))
        plt.plot(avg_t_values, avg_mean_wdiff, label=f"{opt_name} mean(w1-w2)")
        plt.xlabel("Iterations")
        plt.ylabel("Mean(w1 - w2)")
        plt.title(f"Averaged w1-w2 trajectory ({opt_name}, lr={learning_rate})")
        plt.tight_layout()
        plt.legend()
        plt.savefig(wdiff_avg_png_path, dpi=200)
        plt.close()

        other_opt = "gd" if opt_name == "ADAM" else "adam"
        other_avg_path = out / f"experiment_5f_wdiff_avg_{other_opt}_lr{lr_tag}.csv"
        other_curve = _load_avg_curve(other_avg_path)
        if other_curve is not None:
            other_t, other_mean = other_curve
            plt.figure(figsize=(8, 5))
            plt.plot(avg_t_values, avg_mean_wdiff, label=f"{opt_name} mean(w1-w2)")
            plt.plot(other_t, other_mean, label=f"{other_opt.upper()} mean(w1-w2)")
            plt.xlabel("Iterations")
            plt.ylabel("Mean(w1 - w2)")
            plt.title(f"Adam vs GD averaged w1-w2 (lr={learning_rate})")
            plt.tight_layout()
            plt.legend()
            plt.savefig(compare_png_path, dpi=200)
            plt.close()

    adam_bias_counts = None
    adam_bias_edges = None
    if opt_name == "ADAM" and len(adam_bias_abs_values) > 0:
        adam_bias_arr = np.array(adam_bias_abs_values, dtype=float)
        adam_bias_counts, adam_bias_edges = np.histogram(adam_bias_arr, bins=40)

        plt.figure(figsize=(8, 5))
        plt.hist(adam_bias_arr, bins=40)
        plt.title(f"Histogram of |b2-b1| (ADAM, lr={learning_rate})")
        plt.xlabel("|b2 - b1|")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(adam_bias_hist_png_path, dpi=200)
        plt.close()

        with open(adam_bias_hist_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(adam_bias_counts)):
                writer.writerow([
                    adam_bias_edges[i],
                    adam_bias_edges[i + 1],
                    adam_bias_counts[i],
                ])

    # -------------------------
    # Summary file
    # -------------------------
    with open(summary_txt_path, "w") as f:
        f.write("=== Experiment 5f summary ===\n")
        f.write(f"num_runs={num_runs}\n")
        f.write(f"max_iterations={max_iterations}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"optimizer={optimizer_name.upper()}\n")
        f.write(f"tol={tol}, loss_threshold={loss_threshold}\n\n")
        f.write(f"sample_every={sample_every}\n")
        f.write(f"track_weight_diff={track_weight_diff}\n\n")

        f.write(f"Hit condition: {count_hit}/{num_runs}\n")
        f.write(f"Failed (loss-abort): {count_loss_abort}/{num_runs}\n")
        f.write(f"Failed (max-iterations): {count_max_iterations}/{num_runs}\n\n")

        if hist1_counts is not None:
            f.write("Hit-time histogram edges:\n")
            f.write(",".join(map(str, hist1_edges)) + "\n")
            f.write("Hit-time histogram counts:\n")
            f.write(",".join(map(str, hist1_counts)) + "\n\n")

        if hist2_counts is not None:
            f.write("Metric histogram edges:\n")
            f.write(",".join(map(str, hist2_edges)) + "\n")
            f.write("Metric histogram counts:\n")
            f.write(",".join(map(str, hist2_counts)) + "\n")

        if avg_t_values is not None and avg_mean_wdiff is not None and avg_counts is not None:
            f.write("\nAveraged w1-w2 trajectory:\n")
            f.write(f"samples={len(avg_t_values)}\n")
            f.write(f"first_t={int(avg_t_values[0])}, first_mean={float(avg_mean_wdiff[0])}\n")
            f.write(f"last_t={int(avg_t_values[-1])}, last_mean={float(avg_mean_wdiff[-1])}\n")
            f.write(f"last_count={int(avg_counts[-1])}\n")
            f.write(f"trajectory_csv={trajectory_csv_path}\n")
            f.write(f"wdiff_avg_csv={wdiff_avg_csv_path}\n")
            f.write(f"wdiff_avg_plot={wdiff_avg_png_path}\n")
            if Path(compare_png_path).exists():
                f.write(f"wdiff_compare_plot={compare_png_path}\n")

        if adam_bias_counts is not None:
            f.write("\nAdam |b2-b1| histogram:\n")
            f.write("edges:\n")
            f.write(",".join(map(str, adam_bias_edges)) + "\n")
            f.write("counts:\n")
            f.write(",".join(map(str, adam_bias_counts)) + "\n")
            f.write(f"adam_bias_hist_csv={adam_bias_hist_csv_path}\n")
            f.write(f"adam_bias_hist_plot={adam_bias_hist_png_path}\n")

    print("\n=== Experiment 5f summary ===")
    print(f"Hit condition: {count_hit}/{num_runs}")
    print(f"Failed (loss-abort): {count_loss_abort}/{num_runs}")
    print(f"Failed (max-iterations): {count_max_iterations}/{num_runs}")
    print("Files written:")
    print(" - experiment_5f_runs.csv")
    print(" - experiment_5f_summary.txt")
    if hist1_counts is not None:
        print(" - experiment_5f_hit_time_hist.png")
        print(" - experiment_5f_hit_time_hist.csv")
    if hist2_counts is not None:
        print(" - experiment_5f_metric_hist.png")
        print(" - experiment_5f_metric_hist.csv")
    if track_weight_diff and len(trajectory_records) > 0:
        print(f" - {trajectory_csv_path}")
        print(f" - {wdiff_avg_csv_path}")
        print(f" - {wdiff_avg_png_path}")
        if Path(compare_png_path).exists():
            print(f" - {compare_png_path}")
    if adam_bias_counts is not None:
        print(f" - {adam_bias_hist_png_path}")
        print(f" - {adam_bias_hist_csv_path}")



def run_experiment_5f_parallel(
    num_workers: int,
    num_runs: int,
    output_dir: str = ".",
    seed: int = 42,
    max_iterations: int = 10_000_000,
    learning_rate: float = 0.05,
    optimizer_name: str = "gd",
    beta1: float = 0.9,
    beta2: float = 0.999,
    sample_every: int = 1000,
    track_weight_diff: bool = False,
    **kwargs,
):
    """Split num_runs across num_workers subprocesses, each writing to output_dir/worker_N/,
    then merge experiment_5f_runs.csv and regenerate convergence histograms."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_per_worker = num_runs // num_workers
    remainder = num_runs % num_workers

    main_py = Path(__file__).parent.parent / "main.py"

    procs = []
    for i in range(num_workers):
        n = runs_per_worker + (1 if i < remainder else 0)
        worker_out = out / f"worker_{i}"
        cmd = [
            sys.executable, str(main_py),
            "--runs", str(n),
            "--max-iterations", str(max_iterations),
            "--optimizer", optimizer_name,
            "--lr", str(learning_rate),
            "--seed", str(seed + i),
            "--beta1", str(beta1),
            "--beta2", str(beta2),
            "--sample-every", str(sample_every),
            "--output-dir", str(worker_out),
            "--workers", "1",
        ]
        if not track_weight_diff:
            cmd.append("--no-track-weight-diff")
        log_path = out / f"worker_{i}.log"
        log_file = open(log_path, "w")
        procs.append((i, subprocess.Popen(cmd, stdout=log_file, stderr=log_file)))
        print(f"  Worker {i}: {n} runs → {worker_out}")

    print(f"Launched {num_workers} workers for {num_runs} total runs. Waiting...")
    for i, p in procs:
        ret = p.wait()
        print(f"  Worker {i} done (exit {ret})")

    # Merge runs CSVs
    merged_csv = out / "experiment_5f_runs.csv"
    header_written = False
    global_run = 0
    with open(merged_csv, "w", newline="") as fout:
        for i in range(num_workers):
            worker_csv = out / f"worker_{i}" / "experiment_5f_runs.csv"
            if not worker_csv.exists():
                continue
            with open(worker_csv, newline="") as fin:
                reader = csv.DictReader(fin)
                if not header_written:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                else:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                for row in reader:
                    row["run"] = global_run
                    writer.writerow(row)
                    global_run += 1

    # Regenerate convergence histograms from merged CSV
    opt_name = optimizer_name.upper()
    lr = learning_rate
    lr_tag = _format_lr_tag(lr)
    # sample_every already in scope

    deltas, stds = [], []
    with open(merged_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("convergence_delta") not in ("", None):
                deltas.append(float(row["convergence_delta"]))
            if row.get("convergence_std") not in ("", None):
                stds.append(float(row["convergence_std"]))

    for arr, label, xlabel, tag in [
        (deltas, "delta", "|Δ|b2-b1||", "convergence_delta"),
        (stds,   "std",   "std(|b2-b1|)", "convergence_std"),
    ]:
        if not arr:
            continue
        a = np.array(arr)
        counts, edges = np.histogram(a, bins=40)
        png = out / f"experiment_5f_{tag}_hist_{opt_name.lower()}_lr{lr_tag}.png"
        csv_path = out / f"experiment_5f_{tag}_hist_{opt_name.lower()}_lr{lr_tag}.csv"

        plt.figure(figsize=(8, 5))
        plt.hist(a, bins=40)
        title_label = "Option A" if label == "delta" else "Option B"
        plt.title(f"{title_label}: {xlabel} over last {10 * sample_every} iters ({opt_name}, lr={lr})")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(png, dpi=200)
        plt.close()

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin_left", "bin_right", "count"])
            for j in range(len(counts)):
                w.writerow([edges[j], edges[j + 1], counts[j]])

    print(f"\nMerged {global_run} runs → {merged_csv}")
    print(f"Convergence histograms written to {out}/")
def experiment_1d_from_disk_ratio(
    min_dist_pos: float,
    min_dist_neg: float,
    net_val_pos: float,
    net_val_neg: float,
    optimizer_name: str = "gd",
    learning_rate: float = 0.01,
    max_iterations: int = 1_000_000,
    loss_threshold: float = 0.5,
    tol: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    report_every: int = 10_000,
    output_prefix: str = "experiment_1d_disk_ratio",
) -> dict:
    """
    Run a single 1D 2-neuron experiment matching the disk experiment's boundary.

    Constraints satisfied by the initialization:
      - dist(+1, boundary) / dist(-1, boundary) == min_dist_pos / min_dist_neg
      - f(+1) == net_val_pos  (disk network value at the closest positive point)
      - f(-1) == net_val_neg  (disk network value at the closest negative point)

    Derivation (v=[1,-1] fixed, single breakpoint at x*):
      x*   = (1 - ratio) / (1 + ratio),    ratio = min_dist_pos / min_dist_neg
      w1   = net_val_pos / (1 - x*)
      b1   = -w1 * x*
      w2   = net_val_neg / (1 + x*)
      b2   = -w2 * x*

    Saves the same file set as experiment_5f_hit_linear_condition_with_low_loss
    (runs CSV, trajectory CSV, wdiff-avg CSV/PNG, summary TXT, Adam bias CSV/PNG).
    """
    print(
        f"\nexperiment_1d_from_disk_ratio: "
        f"min_dist_pos={min_dist_pos:.6g}, min_dist_neg={min_dist_neg:.6g}, "
        f"net_val_pos={net_val_pos:.6g}, net_val_neg={net_val_neg:.6g}"
    )

    ratio = min_dist_pos / min_dist_neg
    x_star = (1.0 - ratio) / (1.0 + ratio)

    w1_0 = net_val_pos / (1.0 - x_star)
    b1_0 = -w1_0 * x_star
    w2_0 = net_val_neg / (1.0 + x_star)
    b2_0 = -w2_0 * x_star

    print(f"  ratio={ratio:.6g}  x*={x_star:.6g}  w=[{w1_0:.6g}, {w2_0:.6g}]  b=[{b1_0:.6g}, {b2_0:.6g}]")

    opt_name = optimizer_name.upper()
    lr_tag = _format_lr_tag(learning_rate)

    runs_csv_path       = f"{output_prefix}_runs.csv"
    summary_txt_path    = f"{output_prefix}_summary.txt"
    trajectory_csv_path = f"{output_prefix}_wdiff_trajectory_{opt_name.lower()}_lr{lr_tag}.csv"
    wdiff_avg_csv_path  = f"{output_prefix}_wdiff_avg_{opt_name.lower()}_lr{lr_tag}.csv"
    wdiff_avg_png_path  = f"{output_prefix}_wdiff_avg_{opt_name.lower()}_lr{lr_tag}.png"
    hit_time_csv_path   = f"{output_prefix}_hit_time_hist.csv"
    hit_time_png_path   = f"{output_prefix}_hit_time_hist.png"
    adam_bias_csv_path  = f"{output_prefix}_bias_abs_hist_adam_lr{lr_tag}.csv"
    adam_bias_png_path  = f"{output_prefix}_bias_abs_hist_adam_lr{lr_tag}.png"

    params = NetworkParams(
        w=np.array([w1_0, w2_0]),
        b=np.array([b1_0, b2_0]),
        v=np.array([1.0, -1.0]),
    )

    x_data = np.array([1.0, -1.0])
    y_data = np.array([1.0, -1.0])

    optimizer_state = (
        {"beta1": beta1, "beta2": beta2, "eps": eps} if opt_name == "ADAM" else None
    )

    t = 0
    t_hit = -1
    stop_reason = "max-iterations"
    min_vhat_norm = float("inf")
    trajectory_records = []
    sampled_iterations = set()
    loss_t = float("nan")
    diff_history = deque(maxlen=1000) if opt_name == "ADAM" else None

    while t <= max_iterations:
        preds = network_forward(params, x_data)
        loss_t = float(exponential_loss(y_data, preds))

        w1 = float(params.w[0])
        b1 = float(params.b[0])
        w2 = float(params.w[1])
        b2 = float(params.b[1])
        expr_t = abs(w1 + b1 + w2 - b2)

        if t % report_every == 0:
            trajectory_records.append({
                "run": 0,
                "optimizer": opt_name,
                "learning_rate": learning_rate,
                "t": t,
                "w_1": w1,
                "w_2": w2,
                "b_1": b1,
                "b_2": b2,
                "loss": loss_t,
                "stop_reason": "",  # back-filled after loop
            })
            sampled_iterations.add(t)

        if t == 10_000 and loss_t > loss_threshold:
            stop_reason = "loss-abort"
            print(f"  [t={t}] 1D loss {loss_t:.6g} > {loss_threshold:.6g} — aborting.")
            break

        if opt_name == "GD":
            if loss_t < loss_threshold and expr_t < tol:
                t_hit = t
                stop_reason = "hit-condition"
                print(f"  [t={t}] GD hit condition: loss={loss_t:.6g}, |w1+b1+w2-b2|={expr_t:.6g}")
                if t not in sampled_iterations:
                    trajectory_records.append({
                        "run": 0,
                        "optimizer": opt_name,
                        "learning_rate": learning_rate,
                        "t": t,
                        "w_1": w1,
                        "w_2": w2,
                        "b_1": b1,
                        "b_2": b2,
                        "loss": loss_t,
                        "stop_reason": "",
                    })
                break
        elif opt_name == "ADAM":
            assert diff_history is not None
            diff_history.append(abs(b2 - b1))
            if len(diff_history) == diff_history.maxlen:
                std_diff = float(np.std(np.asarray(diff_history)))
                if std_diff < 1e-7:
                    t_hit = t
                    stop_reason = "hit-condition"
                    print(f"  [t={t}] Adam hit condition: std(b2-b1)={std_diff:.6g}")
                    if t not in sampled_iterations:
                        trajectory_records.append({
                            "run": 0,
                            "optimizer": opt_name,
                            "learning_rate": learning_rate,
                            "t": t,
                            "w_1": w1,
                            "w_2": w2,
                            "b_1": b1,
                            "b_2": b2,
                            "loss": loss_t,
                            "stop_reason": "",
                        })
                    break

        if t == max_iterations:
            break

        params, _, adam_info = gradient_descent_step(
            params, x_data, y_data, learning_rate,
            optimizer_name=optimizer_name,
            optimizer_state=optimizer_state,
        )

        if opt_name == "ADAM" and adam_info is not None:
            v_w_hat = adam_info["v_w_hat"]
            v_b_hat = adam_info["v_b_hat"]
            cur_norm = float(np.linalg.norm(np.concatenate([v_w_hat, v_b_hat])))
            min_vhat_norm = min(min_vhat_norm, cur_norm)

        t += 1

    # Back-fill stop_reason in all trajectory rows
    for rec in trajectory_records:
        rec["stop_reason"] = stop_reason

    # Final parameters
    w1_T = float(params.w[0])
    b1_T = float(params.b[0])
    w2_T = float(params.w[1])
    b2_T = float(params.b[1])
    final_bias_abs = abs(b2_T - b1_T)
    expr_last = abs(w1_T + b1_T + w2_T - b2_T)

    # ---- runs CSV ----
    with open(runs_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "stop_reason", "t_last", "t_hit",
            "w1_0", "b1_0", "w2_0", "b2_0",
            "w1_T", "b1_T", "w2_T", "b2_T",
            "loss_last", "expr_last",
            "metric_min", "adam_min_norm_vhat", "final_bias_abs",
        ])
        writer.writeheader()
        writer.writerow({
            "run": 0,
            "stop_reason": stop_reason,
            "t_last": t,
            "t_hit": t_hit,
            "w1_0": w1_0, "b1_0": b1_0, "w2_0": w2_0, "b2_0": b2_0,
            "w1_T": w1_T, "b1_T": b1_T, "w2_T": w2_T, "b2_T": b2_T,
            "loss_last": loss_t,
            "expr_last": expr_last,
            "metric_min": "",
            "adam_min_norm_vhat": float(min_vhat_norm) if min_vhat_norm != float("inf") else "",
            "final_bias_abs": final_bias_abs,
        })

    # ---- trajectory CSV ----
    with open(trajectory_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "optimizer", "learning_rate", "t",
            "w_1", "w_2", "b_1", "b_2", "loss", "stop_reason",
        ])
        writer.writeheader()
        writer.writerows(trajectory_records)

    # ---- wdiff avg CSV + PNG ----
    wdiff_by_t = {rec["t"]: rec["w_1"] - rec["w_2"] for rec in trajectory_records}
    avg_t_values = sorted(wdiff_by_t)

    with open(wdiff_avg_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "mean_w_diff", "std_w_diff", "count"])
        for t_val in avg_t_values:
            writer.writerow([t_val, wdiff_by_t[t_val], 0.0, 1])

    plt.figure(figsize=(8, 5))
    plt.plot(avg_t_values, [wdiff_by_t[tv] for tv in avg_t_values], label=f"{opt_name} w1-w2")
    plt.xlabel("Iterations")
    plt.ylabel("w1 - w2")
    plt.title(f"w1-w2 trajectory ({opt_name}, lr={learning_rate})")
    plt.tight_layout()
    plt.legend()
    plt.savefig(wdiff_avg_png_path, dpi=200)
    plt.close()

    # ---- hit-time histogram CSV + PNG (only if the threshold was reached) ----
    if t_hit >= 0:
        plt.figure(figsize=(8, 5))
        plt.hist([t_hit], bins=1)
        plt.title("Hit time (single run)")
        plt.xlabel("Iterations")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(hit_time_png_path, dpi=200)
        plt.close()

        with open(hit_time_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            writer.writerow([t_hit, t_hit + 1, 1])

    # ---- Adam bias-abs CSV + PNG ----
    if opt_name == "ADAM":
        plt.figure(figsize=(8, 5))
        plt.bar([0], [final_bias_abs], width=0.5)
        plt.title(f"|b2-b1| at convergence (ADAM, lr={learning_rate})")
        plt.xlabel("|b2 - b1|")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(adam_bias_png_path, dpi=200)
        plt.close()

        with open(adam_bias_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            writer.writerow([max(0.0, final_bias_abs - 1e-9), final_bias_abs + 1e-9, 1])

    # ---- summary TXT ----
    with open(summary_txt_path, "w") as f:
        f.write("=== Experiment 1D (from disk ratio) summary ===\n")
        f.write(f"min_dist_pos={min_dist_pos}, min_dist_neg={min_dist_neg}\n")
        f.write(f"ratio={ratio}, x_star={x_star}\n")
        f.write(f"net_val_pos={net_val_pos}, net_val_neg={net_val_neg}\n")
        f.write(f"optimizer={opt_name}, learning_rate={learning_rate}\n")
        f.write(f"loss_threshold={loss_threshold}, tol={tol}\n")
        f.write(f"max_iterations={max_iterations}\n\n")
        f.write(f"num_runs=1\n")
        f.write(f"stop_reason={stop_reason}\n")
        f.write(f"t_last={t}\n")
        f.write(f"t_hit={t_hit}\n\n")
        f.write(f"trajectory_csv={trajectory_csv_path}\n")
        f.write(f"wdiff_avg_csv={wdiff_avg_csv_path}\n")
        if t_hit >= 0:
            f.write(f"hit_time_csv={hit_time_csv_path}\n")
        if opt_name == "ADAM":
            f.write(f"adam_bias_hist_csv={adam_bias_csv_path}\n")

    print(
        f"  stop_reason={stop_reason}, t_hit={t_hit}, "
        f"trajectory_rows={len(trajectory_records)}"
    )
    print(
        f"  Wrote: {runs_csv_path}, {trajectory_csv_path}, "
        f"{wdiff_avg_csv_path}, {summary_txt_path}"
    )

    return {
        "min_dist_pos": min_dist_pos,
        "min_dist_neg": min_dist_neg,
        "ratio": ratio,
        "x_star": x_star,
        "init_w1": w1_0,
        "init_b1": b1_0,
        "init_w2": w2_0,
        "init_b2": b2_0,
        "final_w1": w1_T,
        "final_b1": b1_T,
        "final_w2": w2_T,
        "final_b2": b2_T,
        "stop_reason": stop_reason,
        "t_hit": t_hit,
    }
