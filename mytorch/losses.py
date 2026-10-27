from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable (max subtracted before exp)."""
    shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return shifted / np.sum(shifted, axis=1, keepdims=True)


class Loss(ABC):
    """Scalar loss averaged over the batch.

    forward(y_pred, y_true) caches what backward() needs; backward() returns
    dL/dy_pred, which is fed to the last layer of the network.
    """

    name: str = ""

    def __init__(self) -> None:
        self._cache: Optional[tuple] = None

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def output_transform(self, outputs: np.ndarray) -> np.ndarray:
        """Turn raw network outputs into predictions (identity by default)."""
        return outputs

    def _cached(self) -> tuple:
        if self._cache is None:
            raise RuntimeError(f"{self.__class__.__name__}.backward() called before forward().")
        return self._cache

    @staticmethod
    def _check_shapes(y_pred: np.ndarray, y_true: np.ndarray) -> None:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Loss shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}.")

    def get_config(self) -> Dict[str, Any]:
        return {"type": self.name}


class MSELoss(Loss):
    """Mean squared error over every element: L = mean((y_pred - y_true)^2)."""

    name = "mse"

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._check_shapes(y_pred, y_true)
        diff = y_pred - y_true
        self._cache = (diff,)
        return float(np.mean(diff ** 2))

    def backward(self) -> np.ndarray:
        (diff,) = self._cached()
        return 2.0 * diff / diff.size


class SoftmaxCrossEntropyLoss(Loss):
    """Softmax + cross-entropy on raw logits z and one-hot targets Y.

    L = -1/N * sum(Y * log(softmax(z))). Computed with log-sum-exp, so it never
    takes log(0). The combined gradient simplifies to (softmax(z) - Y) / N.
    """

    name = "cross_entropy"

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._check_shapes(y_pred, y_true)
        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        self._cache = (np.exp(log_probs), y_true)
        return float(-np.sum(y_true * log_probs) / y_pred.shape[0])

    def backward(self) -> np.ndarray:
        probs, y_true = self._cached()
        return (probs - y_true) / probs.shape[0]

    def output_transform(self, outputs: np.ndarray) -> np.ndarray:
        return softmax(outputs)


_LOSSES: Dict[str, Type[Loss]] = {
    "mse": MSELoss,
    "cross_entropy": SoftmaxCrossEntropyLoss,
    "crossentropy": SoftmaxCrossEntropyLoss,
    "softmax_cross_entropy": SoftmaxCrossEntropyLoss,
}


def get_loss(name: str) -> Loss:
    """Return a new loss instance matching `name` or raise ValueError."""
    key = str(name).strip().lower()
    if key not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_LOSSES.keys())}")
    return _LOSSES[key]()
