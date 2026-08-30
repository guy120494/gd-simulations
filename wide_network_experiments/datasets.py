import numpy as np
from typing import Tuple


def create_dataset(
    symmetric: bool = True,
    shift: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the default 2-point dataset.

    symmetric=True -> x=[-1, 1] (with scale+shift), y=[-1, 1]
    symmetric=False -> a simple asymmetric variant for demos.
    """
    if symmetric:
        x = np.array([-1.0, 1.0]) * scale + shift
        y = np.array([-1.0, 1.0])
    else:
        x = np.array([-1.2, 1]) * scale + shift
        y = np.array([-1.0, 1.0])
    return x, y

