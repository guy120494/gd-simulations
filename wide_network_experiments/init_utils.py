import numpy as np
from typing import List, Optional

from .core import NetworkParams

def initialize_network(
    k: int,
    init_type: str = "random",
    w1_init: Optional[float] = None,
    b1_init: Optional[float] = None,
    w2_init: Optional[float] = None,
    b2_init: Optional[float] = None,
    M: Optional[float] = None,
    seed: Optional[int] = None,
    w_init: Optional[np.ndarray] = None,
    b_init: Optional[np.ndarray] = None,
    v_init: Optional[np.ndarray] = None,
    w_binary: Optional[List[int]] = None,
    b_scale: float = 0.1,
    v_binary: bool = False,
) -> NetworkParams:
    """
    Initialization helper.

    init_type:
      - random, symmetric, thesis (k=2), binary
    Or pass explicit arrays via w_init/b_init/v_init.
    """
    if seed is not None:
        np.random.seed(seed)

    # Explicit arrays override init_type.
    if w_init is not None or b_init is not None or v_init is not None:
        if w_init is None:
            w_init = np.random.randn(k) * 0.5
        if b_init is None:
            b_init = np.random.randn(k) * 0.5
        if v_init is None:
            v_init = np.random.choice([-1, 1], size=k) * np.random.rand(k)
        assert len(w_init) == k and len(b_init) == k and len(v_init) == k
        return NetworkParams(w=np.array(w_init), b=np.array(b_init), v=np.array(v_init))

    # Explicit ±1 pattern for w.
    if w_binary is not None:
        assert len(w_binary) == k
        w = np.array(w_binary, dtype=float)
        b = np.random.randn(k) * b_scale
        v = np.random.choice([-1, 1], size=k) if v_binary else np.random.choice([-1, 1], size=k) * np.random.rand(k)
        return NetworkParams(w=w, b=b, v=v.astype(float))

    if init_type == "thesis" and k == 2:
        if w1_init is None:
            w1_init = 1.0
        if b1_init is None:
            b1_init = 1.0

        if w2_init is not None and b2_init is not None:
            w2 = w2_init
            b2 = b2_init
        else:
            if M is None:
                M = 10.0
            w2 = -M
            b2 = M

        return NetworkParams(
            w=np.array([w1_init, w2], dtype=float),
            b=np.array([b1_init, b2], dtype=float),
            v=np.array([1.0, -1.0], dtype=float),
        )

    if init_type == "binary":
        w = np.random.choice([-1, 1], size=k).astype(float)
        b = (np.random.randn(k) * b_scale).astype(float)
        v = np.random.choice([-1, 1], size=k).astype(float) if v_binary else (np.random.choice([-1, 1], size=k) * np.random.rand(k)).astype(float)
        return NetworkParams(w=w, b=b, v=v)

    if init_type == "symmetric":
        return NetworkParams(
            w=(np.random.randn(k) * 0.1).astype(float),
            b=(np.random.randn(k) * 0.1).astype(float),
            v=np.random.choice([-1, 1], size=k).astype(float),
        )

    # random
    return NetworkParams(
        w=(np.random.randn(k) * 0.5).astype(float),
        b=(np.random.randn(k) * 0.5).astype(float),
        v=(np.random.choice([-1, 1], size=k) * np.random.rand(k)).astype(float),
    )

