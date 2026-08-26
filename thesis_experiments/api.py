"""
Public API re-exports.

This module intentionally imports the full stack (including matplotlib via
`experiments.py`). It should not be imported by the CLI fast-path.
"""

from .core import (
    NetworkParams,
    relu,
    network_forward,
    exponential_loss,
    compute_gradients,
    gradient_descent_step,
)

from .init_utils import (
    initialize_network,
)
from .datasets import (
    create_dataset,
)
from .experiments import (
    experiment_5f_hit_linear_condition_with_low_loss,
    run_experiment_5f_parallel,
)

__all__ = [
    # core
    "NetworkParams",
    "relu",
    "network_forward",
    "exponential_loss",
    "compute_gradients",
    "gradient_descent_step",
    "train_gd",
    # metrics
    "find_decision_boundaries",
    "compute_margin",
    "compute_margin_gap",
    # init
    "get_theta_vector",
    "print_initial_params",
    "initialize_network",
    # datasets
    "create_dataset",
    # experiments
    "experiment_1_arbitrary_neurons",
    "experiment_2_boundary_count",
    "experiment_3_robust_case",
    "experiment_4_non_symmetric",
    "experiment_5_overparameterized",
    "experiment_5b_highdimensional_clustered",
    "experiment_5c_margin_convergence_rate",
    "experiment_5d_mixture",
    "example_initialization_options",
]

