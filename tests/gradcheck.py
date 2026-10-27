"""Numerical gradient checking helpers shared by the layer tests.

Principle: we pick a random upstream gradient G and define the scalar loss
L = sum(layer.forward(X) * G). Then dL/dY = G, so layer.backward(G) must
return dL/dX (and fill the parameter gradients). Each analytic gradient is
compared with a central finite difference:

    dL/dx_i ~= (L(x_i + eps) - L(x_i - eps)) / (2 * eps)

Everything runs in float64: float32 precision is too low for a relative
error below 1e-5.
"""
from typing import Callable, List
import numpy as np

from mytorch.layers.base import Layer


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """||a - n|| / (||a|| + ||n||), robust when both gradients are ~0."""
    num = np.linalg.norm(analytic - numeric)
    den = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    return 0.0 if den == 0.0 else float(num / den)


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite difference of the scalar function f() w.r.t. array x (modified in place)."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        loss_plus = f()
        x[idx] = original - eps
        loss_minus = f()
        x[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
    return grad


def check_layer_gradients(layer: Layer, X: np.ndarray, seed: int = 0) -> List[float]:
    """Return the relative errors for dL/dX followed by every parameter gradient.

    X and the layer parameters must already be float64.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(layer.forward(X).shape)

    def loss() -> float:
        return float(np.sum(layer.forward(X) * G))

    layer.forward(X)
    analytic_dX = layer.backward(G)
    analytic_params = [grad.copy() for _, grad in layer.parameters()]

    errors = [relative_error(analytic_dX, numerical_gradient(loss, X))]
    for (param, _), analytic in zip(layer.parameters(), analytic_params):
        errors.append(relative_error(analytic, numerical_gradient(loss, param)))
    return errors
