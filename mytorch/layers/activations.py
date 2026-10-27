from typing import Any, Dict, Optional
import numpy as np

from mytorch.layers.base import Layer


class Activation(Layer):
    """Parameter-free element-wise layer. Subclasses cache what backward() needs."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: Optional[np.ndarray] = None

    def _cached(self) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError(f"{self.__class__.__name__}.backward() called before forward().")
        return self._cache

    def get_config(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__.lower()}


class ReLU(Activation):
    """ReLU(x) = max(0, x)."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._cache = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self._cached() > 0)


class LeakyReLU(Activation):
    """LeakyReLU(x) = x if x > 0 else alpha * x."""

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._cache = inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * np.where(self._cached() > 0, 1.0, self.alpha)

    def get_config(self) -> Dict[str, Any]:
        return {"type": "leakyrelu", "alpha": self.alpha}


class Sigmoid(Activation):
    """Sigmoid(x) = 1 / (1 + exp(-x)). Input clipped to [-50, 50] to avoid exp overflow."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-np.clip(inputs, -50, 50)))
        self._cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        s = self._cached()
        return grad_output * s * (1.0 - s)


class Tanh(Activation):
    """Tanh(x); derivative is 1 - tanh(x)^2."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        out = np.tanh(inputs)
        self._cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - self._cached() ** 2)


class Softmax(Activation):
    """Row-wise softmax, numerically stable (max subtracted before exp).

    Do NOT put this layer before SoftmaxCrossEntropyLoss: that loss already
    applies softmax to raw logits. Use it only to output probabilities.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        out = shifted / np.sum(shifted, axis=1, keepdims=True)
        self._cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Jacobian-vector product per row: s * (g - <g, s>)
        s = self._cached()
        return s * (grad_output - np.sum(grad_output * s, axis=1, keepdims=True))
