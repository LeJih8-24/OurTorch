from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type
import numpy as np

Parameters = List[Tuple[np.ndarray, np.ndarray]]


class Optimizer(ABC):
    """Updates parameters in place from the (param, grad) pairs given by Layer.parameters().

    Per-parameter state (momentum, Adam moments) is keyed by id(param): this
    relies on layers updating their arrays in place (see the layer conventions).

    weight_decay adds the L2 penalty gradient (weight_decay * param) to every
    gradient before the update, i.e. it minimizes L + weight_decay / 2 * ||W||^2.
    """

    name: str = ""

    def __init__(self, lr: float, weight_decay: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}.")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def _regularized(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.weight_decay == 0.0:
            return grad
        return grad + self.weight_decay * param

    @abstractmethod
    def step(self, parameters: Parameters) -> None:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {"type": self.name, "lr": self.lr, "weight_decay": self.weight_decay}


class SGD(Optimizer):
    """Stochastic gradient descent with (classical) momentum.

        v <- momentum * v - lr * g
        p <- p + v
    momentum=0 gives plain SGD: p <- p - lr * g.
    """

    name = "sgd"

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(lr, weight_decay)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
        self.momentum = float(momentum)
        self._velocity: Dict[int, np.ndarray] = {}

    def step(self, parameters: Parameters) -> None:
        for param, grad in parameters:
            grad = self._regularized(param, grad)
            if self.momentum == 0.0:
                param -= self.lr * grad
                continue
            v = self._velocity.setdefault(id(param), np.zeros_like(param))
            v *= self.momentum
            v -= self.lr * grad
            param += v

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "momentum": self.momentum}


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction.

        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g^2
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    where m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t).
    """

    name = "adam"

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr, weight_decay)
        for label, beta in (("beta1", beta1), ("beta2", beta2)):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{label} must be in [0, 1), got {beta}.")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}

    def step(self, parameters: Parameters) -> None:
        self.t += 1
        correction1 = 1.0 - self.beta1 ** self.t
        correction2 = 1.0 - self.beta2 ** self.t
        for param, grad in parameters:
            grad = self._regularized(param, grad)
            m = self._m.setdefault(id(param), np.zeros_like(param))
            v = self._v.setdefault(id(param), np.zeros_like(param))
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad ** 2
            param -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps}


_OPTIMIZERS: Dict[str, Type[Optimizer]] = {"sgd": SGD, "adam": Adam}


def get_optimizer(config: Dict[str, Any]) -> Optimizer:
    """Build an optimizer from a .conf entry such as {"type": "adam", "lr": 0.001}."""
    if not isinstance(config, dict) or "type" not in config:
        raise ValueError(f"Optimizer config must be a dict with a 'type' key, got {config!r}.")
    params = dict(config)
    key = str(params.pop("type")).strip().lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{key}'. Available: {sorted(_OPTIMIZERS.keys())}")
    try:
        return _OPTIMIZERS[key](**params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for optimizer '{key}': {exc}") from exc
