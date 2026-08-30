"""
Dataset generation for the 2-disk experiment.

Two disks in R^d, one per class:
  Positive (+1): center (1, 0, ..., 0), radius sqrt(d)
  Negative (-1): center (-1, 0, ..., 0), radius sqrt(d)

The first coordinate is fixed at ±1.
The remaining d-1 coordinates are drawn uniformly from
the ball of radius sqrt(d) in R^(d-1).
"""

import numpy as np


def _sample_unit_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample n points from the unit ball in R^d."""
    raw = rng.standard_normal((n, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit_directions = raw / norms
    radii = rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / d)
    return unit_directions * radii


def create_disk_dataset(n: int, d: int, seed: int = 42):
    """
    Sample n points total from two d-dimensional disks.

    n_pos = n // 2 positive samples, n_neg = n - n_pos negative.

    Returns
    -------
    X : (n, d) array
    y : (n,) array with values ±1
    """
    if d < 1:
        raise ValueError("d must be >= 1")

    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    radius = np.sqrt(d)

    if d == 1:
        X_pos = np.ones((n_pos, 1))
        X_neg = -np.ones((n_neg, 1))
    else:
        rest_pos = _sample_unit_ball(n_pos, d - 1, rng) * radius
        rest_neg = _sample_unit_ball(n_neg, d - 1, rng) * radius
        X_pos = np.concatenate([np.ones((n_pos, 1)), rest_pos], axis=1)
        X_neg = np.concatenate([-np.ones((n_neg, 1)), rest_neg], axis=1)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    return X, y
