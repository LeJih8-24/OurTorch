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

    # Names of the array attributes that must be saved with the network
    # (trainable weights and fixed statistics). Empty for stateless layers.
    state_keys: Tuple[str, ...] = ()

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return the arrays to save in a network file, keyed by attribute name."""
        return {key: getattr(self, key) for key in self.state_keys}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        """Copy saved arrays into the layer, in place (optimizer references stay valid).

        Raises ValueError on missing/unexpected keys or shape mismatch.
        """
        if set(state) != set(self.state_keys):
            raise ValueError(
                f"{self.__class__.__name__} state keys {sorted(state)} do not match {sorted(self.state_keys)}."
            )
        for key in self.state_keys:
            current = getattr(self, key)
            value = np.asarray(state[key])
            if value.shape != current.shape:
                raise ValueError(
                    f"{self.__class__.__name__}.{key} has shape {value.shape}, expected {current.shape}."
                )
            current[...] = value
