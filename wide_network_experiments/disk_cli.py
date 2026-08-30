import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a 2-layer ReLU network on the 2-disk dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_disks.py
  python main_disks.py --n 20 --d 5 --k 5 --lr 0.1
  python main_disks.py --optimizer adam --beta1 0.9 --beta2 0.999
  python main_disks.py --num-runs 100 --seed 0
""",
    )

    # Dataset
    parser.add_argument("--n",  type=int, default=100, help="Total training samples (default: 100)")
    parser.add_argument("--d",  type=int, default=10,  help="Input dimension (default: 10)")
    parser.add_argument("--k",  type=int, default=4,   help="Network width (default: 4)")

    # Training
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=0.01,  help="Learning rate (default: 0.01)")
    parser.add_argument("--max-iterations",         type=int,   default=1_000_000,        help="Max Phase-2 steps per run (default: 1 000 000)")
    parser.add_argument("--report-every",           type=int,   default=10_000,           help="Boundary-distance checkpoint interval (default: 10 000)")
    parser.add_argument("--seed",  "-s",            type=int,   default=42,               help="Base random seed (default: 42)")
    parser.add_argument("--num-runs",               type=int,   default=1,                help="Number of independent runs (default: 1)")

    # Optimizer
    parser.add_argument("--optimizer", choices=["gd", "adam", "GD", "ADAM"], default="gd", help="Optimizer (default: gd)")
    parser.add_argument("--beta1",     type=float, default=0.9,   help="Adam beta1 (default: 0.9)")
    parser.add_argument("--beta2",     type=float, default=0.999, help="Adam beta2 (default: 0.999)")

    # Output
    parser.add_argument("--output-csv",          default=None)
    parser.add_argument("--dist-comparison-csv", default=None)
    parser.add_argument("--dist-comparison-png", default="")
    parser.add_argument("--weight-diff-csv",     default=None)
    parser.add_argument("--unified-comparison-csv", default=None,
                        help="(run, step, min_dist_large, min_dist_small) on a single "
                             "absolute step axis spanning Phase 1 and Phase 2")

    args = parser.parse_args(argv)

    tag = f"{args.optimizer.lower()}_k{args.k}_lr{args.lr}_n{args.n}_d{args.d}_runs{args.num_runs}"
    if args.output_csv is None:
        args.output_csv = f"experiment_disks_summary_{tag}.csv"
    if args.dist_comparison_csv is None:
        args.dist_comparison_csv = f"experiment_disks_dist_comparison_{tag}.csv"
    if args.weight_diff_csv is None:
        args.weight_diff_csv = f"experiment_disks_weight_diff_{tag}.csv"
    if args.unified_comparison_csv is None:
        args.unified_comparison_csv = f"experiment_disks_unified_comparison_{tag}.csv"

    print("=" * 60)
    print("Disk Experiment — 2-layer ReLU network on 2-disk dataset")
    print("=" * 60)
    print(args)
    print()

    from .experiment_disks import experiment_disks

    experiment_disks(
        n=args.n,
        d=args.d,
        k=args.k,
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        max_iterations=args.max_iterations,
        report_every=args.report_every,
        seed=args.seed,
        num_runs=args.num_runs,
        beta1=args.beta1,
        beta2=args.beta2,
        output_csv=args.output_csv,
        dist_comparison_csv=args.dist_comparison_csv,
        dist_comparison_png=args.dist_comparison_png,
        weight_diff_csv=args.weight_diff_csv,
        unified_comparison_csv=args.unified_comparison_csv,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
