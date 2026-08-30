"""
Backfill Phase-1 margin checkpoints and merge them with an already-computed
Phase-2 dist-comparison CSV to build a unified (absolute-step) comparison CSV,
without rerunning the expensive Phase-2 loop.

Phase 1's dataset and initialization are fully determined by (optimizer, n, d,
k, lr, seed, run_idx), and its GD/Adam updates are deterministic given that
starting point, so rerunning train_phase1 with the same configuration used to
produce an existing Phase-2 CSV reproduces that run's Phase 1 trajectory
exactly (bit-for-bit) -- including the state Phase 2 was actually seeded from.
That means the new Phase-1 checkpoints can be spliced directly onto the
existing Phase-2 rows instead of rerunning Phase 2 from scratch.

Usage
-----
  python backfill_phase1_unified.py --optimizer gd \
      --dist-comparison-csv experiment_disks_dist_comparison_gd_k10_lr0.1_n20_d5_runs200.csv \
      --output-csv experiment_disks_unified_comparison_gd_k10_lr0.1_n20_d5_runs200.csv
"""

import argparse
import csv
import os

import numpy as np

from wide_network_experiments.disk_dataset import create_disk_dataset
from wide_network_experiments.disk_trainer import train_phase1

FIELDNAMES = ["run", "step", "min_dist_large", "min_dist_small"]


def load_phase2_rows(dist_comparison_csv: str) -> dict:
    """Return {run_idx: [{"step", "min_dist_large", "min_dist_small"}, ...]}."""
    by_run: dict = {}
    with open(dist_comparison_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = int(row["run"])
            by_run.setdefault(run, []).append({
                "step": int(row["step"]),
                "min_dist_large": row["min_dist_large"],
                "min_dist_small": row["min_dist_small"],
            })
    return by_run


def load_completed_runs(output_csv: str) -> set:
    """Run indices already written to output_csv from a prior (possibly interrupted) pass."""
    if not os.path.exists(output_csv):
        return set()
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["run"]) for row in reader}


def backfill(
    *,
    optimizer_name: str,
    n: int,
    d: int,
    k: int,
    learning_rate: float,
    max_iterations: int,
    report_every: int,
    seed: int,
    beta1: float,
    beta2: float,
    eps: float,
    dist_comparison_csv: str,
    output_csv: str,
) -> None:
    loss_threshold = 1.0 / n
    phase2_by_run = load_phase2_rows(dist_comparison_csv)
    run_ids = sorted(phase2_by_run)
    print(f"{optimizer_name}: {len(run_ids)} successful runs found in {dist_comparison_csv}")

    completed = load_completed_runs(output_csv)
    todo = [r for r in run_ids if r not in completed]
    if completed:
        print(f"Resuming: {len(completed)} runs already in {output_csv}, "
              f"{len(todo)} remaining.")

    # Write the header only if the file doesn't already exist (fresh start).
    file_exists = os.path.exists(output_csv)
    f = open(output_csv, "a" if file_exists else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        f.flush()

    merged_count = 0
    try:
        for run_idx in todo:
            run_seed = seed + run_idx
            X, y = create_disk_dataset(n, d, seed=run_seed)
            rng = np.random.default_rng(run_seed + 1)
            W = rng.standard_normal((k, d)) * np.sqrt(2.0 / d)
            b = np.zeros(k)
            v = np.concatenate([np.ones(k // 2), -np.ones(k - k // 2)])

            W, b, v, t_threshold, adam_state, reason, checkpoints = train_phase1(
                W, b, v, X, y,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                loss_threshold=loss_threshold,
                train_v=False,
                beta1=beta1, beta2=beta2, eps=eps,
                report_every=report_every,
            )

            if reason != "threshold-reached":
                print(f"  run {run_idx}: WARNING expected 'threshold-reached', got {reason!r} "
                      f"-- this run's Phase 1 no longer matches its existing Phase-2 data; skipping.")
                continue
            if t_threshold != max_iterations:
                print(f"  run {run_idx}: NOTE t_threshold={t_threshold} != max_iterations={max_iterations}; "
                      f"using the actual t_threshold as the Phase-2 step offset.")

            run_rows = []
            for cp in checkpoints:
                run_rows.append({
                    "run": run_idx, "step": cp["step"],
                    "min_dist_large": cp["min_dist_large"],
                    "min_dist_small": "",
                })
            for row in phase2_by_run[run_idx]:
                run_rows.append({
                    "run": run_idx, "step": t_threshold + row["step"],
                    "min_dist_large": row["min_dist_large"],
                    "min_dist_small": row["min_dist_small"],
                })

            # Write and flush this run's rows immediately so a crash loses at
            # most the in-progress run, not the whole job.
            writer.writerows(run_rows)
            f.flush()
            os.fsync(f.fileno())
            merged_count += 1

            print(f"  run {run_idx}: merged ({len(checkpoints)} Phase-1 + "
                  f"{len(phase2_by_run[run_idx])} Phase-2 rows) [{merged_count}/{len(todo)} this pass]")
    finally:
        f.close()

    total_merged = len(completed) + merged_count
    print(f"Done. {output_csv}: {total_merged}/{len(run_ids)} runs merged "
          f"({merged_count} in this pass).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizer", required=True, choices=["gd", "adam", "GD", "ADAM"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=1_000_001)
    parser.add_argument("--report-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dist-comparison-csv", required=True,
                        help="Existing Phase-2-relative dist-comparison CSV to splice onto")
    parser.add_argument("--output-csv", required=True,
                        help="Output unified (absolute-step) comparison CSV")
    args = parser.parse_args()

    backfill(
        optimizer_name=args.optimizer.lower(),
        n=args.n, d=args.d, k=args.k,
        learning_rate=args.lr,
        max_iterations=args.max_iterations,
        report_every=args.report_every,
        seed=args.seed,
        beta1=args.beta1, beta2=args.beta2, eps=args.eps,
        dist_comparison_csv=args.dist_comparison_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
