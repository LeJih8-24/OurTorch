from typing import Any, Dict, Optional
import numpy as np

from mytorch.layers.base import Layer


class Normalization(Layer):
    """Input standardization: Y = (X - mean) / std, feature by feature.

    mean and std are NOT trained by the optimizer: they are computed once on the
    training set with adapt(X), then saved with the network so that prediction
    applies exactly the same transformation as training.

    Features whose std is ~0 on the training set (e.g. a square that is always
    empty) keep a std of 1: they are only centered, never divided by ~0.

    Before adapt() the layer is the identity (mean 0, std 1). `count` stores the
    number of samples used by adapt() and is saved too: count == 0 means the
    statistics still have to be computed.
    """

    state_keys = ("mean", "std", "count")
    MIN_STD = 1e-6

    def __init__(self, features: int) -> None:
        super().__init__()
        if int(features) <= 0:
            raise ValueError(f"Normalization features must be positive, got {features}.")
        self.features = int(features)
        self.mean = np.zeros((1, self.features), dtype=np.float32)
        self.std = np.ones((1, self.features), dtype=np.float32)
        self.count = np.zeros((1,), dtype=np.int64)
        self._cache_std: Optional[np.ndarray] = None

    @property
    def adapted(self) -> bool:
        return int(self.count[0]) > 0

    def _check(self, inputs: np.ndarray) -> None:
        if inputs.ndim != 2 or inputs.shape[1] != self.features:
            raise ValueError(
                f"Normalization expected input of shape (batch, {self.features}), got {inputs.shape}."
            )

    def adapt(self, inputs: np.ndarray) -> None:
        """Compute mean and std of each feature on (training) data."""
        self._check(inputs)
        if len(inputs) == 0:
            raise ValueError("Cannot adapt Normalization on an empty dataset.")
        std = inputs.std(axis=0, keepdims=True)
        self.mean[...] = inputs.mean(axis=0, keepdims=True)
        self.std[...] = np.where(std < self.MIN_STD, 1.0, std)
        self.count[0] = len(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._check(inputs)
        self._cache_std = self.std
        return (inputs - self.mean) / self.std

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_std is None:
            raise RuntimeError("Normalization.backward() called before forward().")
        return grad_output / self._cache_std

    def get_config(self) -> Dict[str, Any]:
        return {"type": "normalization", "features": self.features}
