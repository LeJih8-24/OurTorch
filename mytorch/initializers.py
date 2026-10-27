from typing import Callable, Dict, Tuple
import numpy as np


def _validate_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"Expected a 2D shape tuple (n_in, n_out), got {shape!r}.")
    n_in, n_out = int(shape[0]), int(shape[1])
    if n_in <= 0 or n_out <= 0:
        raise ValueError(f"Shape dimensions must be positive, got {shape!r}.")
    return n_in, n_out


def xavier_normal(shape: Tuple[int, int]) -> np.ndarray:
    """Xavier / Glorot normal initialization: N(0, sqrt(2 / (n_in + n_out)))."""
    n_in, n_out = _validate_shape(shape)
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0.0, std, size=(n_in, n_out)).astype(np.float32)


def xavier_uniform(shape: Tuple[int, int]) -> np.ndarray:
    """Xavier / Glorot uniform initialization: U(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))."""
    n_in, n_out = _validate_shape(shape)
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out)).astype(np.float32)


def he_normal(shape: Tuple[int, int]) -> np.ndarray:
    """He / Kaiming normal initialization: N(0, sqrt(2 / n_in))."""
    n_in, n_out = _validate_shape(shape)
    std = np.sqrt(2.0 / n_in)
    return np.random.normal(0.0, std, size=(n_in, n_out)).astype(np.float32)


def lecun_normal(shape: Tuple[int, int]) -> np.ndarray:
    """LeCun normal initialization: N(0, sqrt(1 / n_in))."""
    n_in, n_out = _validate_shape(shape)
    std = np.sqrt(1.0 / n_in)
    return np.random.normal(0.0, std, size=(n_in, n_out)).astype(np.float32)


_INITIALIZERS: Dict[str, Callable[[Tuple[int, int]], np.ndarray]] = {
    # Short aliases used in .conf files (normal variants by default)
    "xavier": xavier_normal,
    "glorot": xavier_normal,
    "he": he_normal,
    "kaiming": he_normal,
    "lecun": lecun_normal,
    "xavier_normal": xavier_normal,
    "glorot_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "glorot_uniform": xavier_uniform,
    "he_normal": he_normal,
    "kaiming_normal": he_normal,
    "lecun_normal": lecun_normal,
}


def get_initializer(name: str) -> Callable[[Tuple[int, int]], np.ndarray]:
    """Return the initializer function matching `name` or raise ValueError."""
    if not isinstance(name, str):
        raise ValueError(f"Initializer name must be a string, got {type(name).__name__}.")
    key = name.strip().lower()
    if key not in _INITIALIZERS:
        raise ValueError(
            f"Unknown initializer '{name}'. Available: {sorted(_INITIALIZERS.keys())}"
        )
    return _INITIALIZERS[key]
