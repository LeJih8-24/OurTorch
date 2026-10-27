from typing import Any, Dict, Optional
import numpy as np

from mytorch.layers.base import Layer


class Dropout(Layer):
    """Inverted dropout: during training each unit is zeroed with probability `rate`
    and the survivors are scaled by 1 / (1 - rate), so no rescaling is needed at
    inference time.

    mc_dropout=True keeps dropout active even when `training` is False (Monte Carlo
    dropout): averaging several stochastic predictions gives a mean prediction and
    an uncertainty estimate.

    The mask uses NumPy's global RNG, so np.random.seed() makes it reproducible.
    """

    def __init__(self, rate: float = 0.2, mc_dropout: bool = False) -> None:
        super().__init__()
        if not 0.0 <= float(rate) < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}.")
        self.rate = float(rate)
        self.mc_dropout = bool(mc_dropout)
        self._mask: Optional[np.ndarray] = None

    @property
    def active(self) -> bool:
        return (self.training or self.mc_dropout) and self.rate > 0.0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.active:
            self._mask = None
            return inputs
        keep = np.random.random(inputs.shape) >= self.rate
        self._mask = keep.astype(inputs.dtype) / (1.0 - self.rate)
        return inputs * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Same mask as the forward pass: dropped units get no gradient.
        if self._mask is None:
            return grad_output
        return grad_output * self._mask

    def get_config(self) -> Dict[str, Any]:
        return {"type": "dropout", "rate": self.rate, "mc_dropout": self.mc_dropout}
