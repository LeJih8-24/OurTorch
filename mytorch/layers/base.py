from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np


class Layer(ABC):
    """Abstract base class for all neural network layers."""

    def __init__(self) -> None:
        self.training: bool = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the layer."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute the backward pass of the layer."""
        raise NotImplementedError

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (param, grad) tuples for trainable parameters."""
        return []

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary for the layer."""
        return {"type": self.__class__.__name__}
