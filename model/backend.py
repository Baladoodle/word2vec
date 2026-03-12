from __future__ import annotations

from config import Config

try:
    import numpy as _np
except Exception:
    _np = None

_cp = None
if Config.use_cupy:
    try:
        import cupy as _cp
    except Exception as exc:
        raise ImportError(
            "CuPy not available"
        ) from exc

xp = _cp if _cp is not None else _np

def to_numpy(x):
    """Convert CuPy arrays to NumPy"""
    if _cp is not None and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return x

def rng(seed: int | None):
    """CuPy adapter for rng"""
    if _cp is not None:
        return _cp.random.RandomState(seed)
    return _np.random.default_rng(seed)
