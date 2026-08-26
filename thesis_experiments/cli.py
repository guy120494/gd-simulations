import argparse
import sys
from typing import List, Optional, Union


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Thesis Experiments: GD Convergence and Adversarial Robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    # Common parameters
    parser.add_argument("--iterations", "-i", type=int, help="Number of GD iterations")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--runs", type=int, help="Number of runs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, help="Max iterations")
    parser.add_argument(
        "--optimizer",
        choices=["gd", "adam", "GD", "ADAM"],
        default="gd",
        help="Optimizer to use: gd or adam",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (momentum decay)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (RMS decay)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1000,
        help="Record w1-w2 trajectory every N iterations (default: 1000)",
    )
    parser.add_argument(
        "--track-weight-diff",
        dest="track_weight_diff",
        action="store_true",
        default=True,
        help="Enable recording/plotting of averaged w1-w2 trajectories (default: enabled)",
    )
    parser.add_argument(
        "--no-track-weight-diff",
        dest="track_weight_diff",
        action="store_false",
        help="Disable recording/plotting of averaged w1-w2 trajectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write output files (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to split runs across (default: 1)",
    )

    args = parser.parse_args(argv)
    print(args)

    # Delayed import so `--help`/`--list-experiments` are fast.
    from .experiments import experiment_5f_hit_linear_condition_with_low_loss, run_experiment_5f_parallel

    print("=" * 60)
    print("Thesis Experiments: GD Convergence and Adversarial Robustness")
    print("=" * 60)

    num_runs = args.runs if args.runs is not None else 10_000
    workers = args.workers

    common_kwargs = dict(
        num_runs=num_runs,
        max_iterations=args.max_iterations if args.max_iterations is not None else 100_000,
        learning_rate=args.lr if args.lr is not None else 0.01,
        optimizer_name=args.optimizer,
        seed=args.seed if args.seed else 42,
        beta1=args.beta1,
        beta2=args.beta2,
        sample_every=args.sample_every,
        track_weight_diff=args.track_weight_diff,
        output_dir=args.output_dir,
    )

    if workers > 1:
        run_experiment_5f_parallel(num_workers=workers, **common_kwargs)
    else:
        experiment_5f_hit_linear_condition_with_low_loss(**common_kwargs)


    print("\n" + "=" * 60)
    print("All requested experiments completed!")
    print("=" * 60)
