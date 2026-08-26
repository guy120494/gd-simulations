"""
Run after killing the parallel experiment workers to merge whatever data
was written and regenerate all histograms.

Usage:
    python merge_results.py adam_conv_lr0p05
    python merge_results.py adam_conv_lr0p01
    python merge_results.py adam_conv_lr0p05 adam_conv_lr0p01
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def merge_and_plot(output_dir: str):
    out = Path(output_dir)
    worker_dirs = sorted(out.glob("worker_*"))

    if not worker_dirs:
        print(f"No worker_* directories found in {out}")
        return

    # --- Merge runs CSVs ---
    merged_csv = out / "experiment_5f_runs.csv"
    all_rows = []
    fieldnames = None
    total = 0

    for wd in worker_dirs:
        csv_path = wd / "experiment_5f_runs.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            print(f"  {wd.name}: no data yet, skipping")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows = list(reader)
            print(f"  {wd.name}: {len(rows)} runs")
            all_rows.extend(rows)
            total += len(rows)

    if not all_rows or fieldnames is None:
        print("No data to merge.")
        return

    # Re-number runs globally
    for i, row in enumerate(all_rows):
        row["run"] = i

    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMerged {total} runs → {merged_csv}")

    # --- Detect optimizer and lr from first row ---
    sample = all_rows[0]
    # Try to infer from directory name
    opt_name = "ADAM" if "adam" in output_dir.lower() else "GD"
    lr_str = output_dir.split("lr")[-1] if "lr" in output_dir else "unknown"
    lr_str_display = lr_str.replace("p", ".")

    def _hist(values, title, xlabel, path_stem):
        if not values:
            return
        arr = np.array(values)
        plt.figure(figsize=(8, 5))
        plt.hist(arr, bins=40)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        png = out / f"{path_stem}.png"
        csv_p = out / f"{path_stem}.csv"
        plt.savefig(png, dpi=200)
        plt.close()
        counts, edges = np.histogram(arr, bins=40)
        with open(csv_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(counts)):
                w.writerow([edges[i], edges[i+1], counts[i]])
        print(f"  Saved {png.name} and {csv_p.name}")

    # --- Convergence delta (Option A) ---
    deltas = [float(r["convergence_delta"]) for r in all_rows
              if r.get("convergence_delta") not in ("", None)]
    _hist(
        deltas,
        f"Option A: |Δ|b2-b1|| over last 10k iters ({opt_name}, lr={lr_str_display})",
        "|Δ|b2-b1||",
        f"experiment_5f_convergence_delta_hist_{opt_name.lower()}_lr{lr_str}",
    )

    # --- Convergence std (Option B) ---
    stds = [float(r["convergence_std"]) for r in all_rows
            if r.get("convergence_std") not in ("", None)]
    _hist(
        stds,
        f"Option B: std(|b2-b1|) over last 10k iters ({opt_name}, lr={lr_str_display})",
        "std(|b2-b1|)",
        f"experiment_5f_convergence_std_hist_{opt_name.lower()}_lr{lr_str}",
    )

    # --- Final |b2-b1| histogram ---
    bias_abs = [float(r["final_bias_abs"]) for r in all_rows
                if r.get("final_bias_abs") not in ("", None)]
    _hist(
        bias_abs,
        f"|b2-b1| at end of run ({opt_name}, lr={lr_str_display})",
        "|b2 - b1|",
        f"experiment_5f_bias_abs_hist_{opt_name.lower()}_lr{lr_str}",
    )

    # --- Summary ---
    n_hit = sum(1 for r in all_rows if r.get("stop_reason") == "hit condition")
    n_abort = sum(1 for r in all_rows if r.get("stop_reason") == "loss-abort")
    print(f"\nSummary for {output_dir}:")
    print(f"  Total runs:   {total}")
    print(f"  Hit (10M):    {n_hit}")
    print(f"  Loss-aborted: {n_abort}")
    if deltas:
        print(f"  Delta median: {np.median(deltas):.3e}  max: {np.max(deltas):.3e}")
    if stds:
        print(f"  Std   median: {np.median(stds):.3e}  max: {np.max(stds):.3e}")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["adam_conv_lr0p05", "adam_conv_lr0p01"]
    for d in dirs:
        print(f"\n=== {d} ===")
        merge_and_plot(d)
