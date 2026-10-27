from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mytorch.initializers import get_initializer
from mytorch.layers.base import Layer


class Linear(Layer):
    """Fully connected layer: Y = X @ W + b.

    Shapes:
        X: (batch, in_features)
        W: (in_features, out_features)
        b: (1, out_features)
        Y: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he") -> None:
        super().__init__()
        if int(in_features) <= 0 or int(out_features) <= 0:
            raise ValueError(
                f"Linear dimensions must be positive, got ({in_features}, {out_features})."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init = init
        self.W: np.ndarray = get_initializer(init)((self.in_features, self.out_features))
        self.b: np.ndarray = np.zeros((1, self.out_features), dtype=self.W.dtype)
        # Gradients are allocated once and updated in place, so the references
        # returned by parameters() stay valid for the optimizer across steps.
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)
        self._cache_X: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected input of shape (batch, {self.in_features}), got {inputs.shape}."
            )
        self._cache_X = inputs
        return inputs @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_X is None:
            raise RuntimeError("Linear.backward() called before forward().")
        self.dW[...] = self._cache_X.T @ grad_output
        self.db[...] = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "in": self.in_features,
            "out": self.out_features,
            "init": self.init,
        }
