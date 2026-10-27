import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mytorch.layers.base import Layer
from mytorch.layers.normalization import Normalization
from mytorch.losses import Loss
from mytorch.optimizers import Optimizer, Parameters

History = Dict[str, Any]


class Sequential:
    """Stack of layers trained with a loss and an optimizer.

    Inputs X have shape (N, n_features). Targets y have shape (N, n_outputs):
    one-hot vectors for classification, in the same order as `classes`.
    Randomness (shuffling, validation split, dropout) uses NumPy's global RNG,
    so np.random.seed() makes a training run reproducible.

    A Normalization layer is only allowed in first position: its statistics are
    computed on the raw training inputs by fit().

    `training_config` (epochs, batch_size, val_split, patience) and `seed` come
    from the .conf file; they are kept so the network can be saved and retrained
    with the same settings: model.fit(X, y, **model.training_config).
    """

    def __init__(
        self,
        layers: List[Layer],
        loss: Loss,
        optimizer: Optimizer,
        classes: List[str],
        training_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not layers:
            raise ValueError("Sequential needs at least one layer.")
        if any(isinstance(layer, Normalization) for layer in layers[1:]):
            raise ValueError("A Normalization layer is only allowed as the first layer.")
        self.layers = list(layers)
        self.loss = loss
        self.optimizer = optimizer
        self.classes = list(classes)
        self.training_config: Dict[str, Any] = dict(training_config or {})
        self.seed = seed

    # ----- modes ---------------------------------------------------------------

    def train(self) -> None:
        """Training mode: Dropout is active."""
        for layer in self.layers:
            layer.training = True

    def eval(self) -> None:
        """Inference mode: Dropout is the identity (unless MC dropout)."""
        for layer in self.layers:
            layer.training = False

    # ----- forward / backward --------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> Parameters:
        return [pair for layer in self.layers for pair in layer.parameters()]

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """One gradient descent step on a mini-batch; returns the batch loss."""
        value = self.loss.forward(self.forward(X_batch), y_batch)
        self.backward(self.loss.backward())
        self.optimizer.step(self.parameters())
        return value

    # ----- training ------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        val_split: float = 0.15,
        patience: Optional[int] = 5,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> History:
        """Mini-batch training with optional validation and early stopping.

        Validation set: `validation_data` if given, otherwise a random
        `val_split` fraction of (X, y) held out from training.
        Early stopping: if the validation loss has not improved for `patience`
        epochs, training stops and the best weights are restored
        (patience=None disables it). Without validation data there is no
        early stopping.
        If the first layer is a Normalization that was never adapted, its
        statistics are computed here on the training part of the data.

        Returns a history dict: train_loss, train_acc, val_loss, val_acc
        (one value per epoch) and best_epoch (1-based, None without validation).
        """
        self._check_xy(X, y)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError(f"epochs and batch_size must be positive, got {epochs} and {batch_size}.")
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {val_split}.")

        if validation_data is not None:
            X_val, y_val = validation_data
            self._check_xy(X_val, y_val)
            X_train, y_train = X, y
        elif val_split > 0.0:
            order = np.random.permutation(len(X))
            n_val = max(1, int(round(len(X) * val_split)))
            if n_val >= len(X):
                raise ValueError("Not enough samples to hold out a validation set.")
            X_val, y_val = X[order[:n_val]], y[order[:n_val]]
            X_train, y_train = X[order[n_val:]], y[order[n_val:]]
        else:
            X_val = y_val = None
            X_train, y_train = X, y

        # Input statistics come from the training part only (never from validation
        # data, which would leak information). They are computed once: retraining a
        # loaded network keeps the statistics it was saved with.
        first = self.layers[0]
        if isinstance(first, Normalization) and not first.adapted:
            first.adapt(X_train)

        history: History = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": None}
        best_val = np.inf
        best_weights: Optional[List[np.ndarray]] = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            self.train()
            order = np.random.permutation(len(X_train))
            total = 0.0
            for start in range(0, len(X_train), batch_size):
                idx = order[start:start + batch_size]
                total += self.train_step(X_train[idx], y_train[idx]) * len(idx)
            history["train_loss"].append(total / len(X_train))
            history["train_acc"].append(self.evaluate(X_train, y_train, batch_size)[1])

            message = f"epoch {epoch}/{epochs} - loss {history['train_loss'][-1]:.4f} - acc {history['train_acc'][-1]:.4f}"
            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                message += f" - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    best_weights = [param.copy() for param, _ in self.parameters()]
                    history["best_epoch"] = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if verbose:
                # Logs go to stderr: stdout is reserved for predictions.
                print(message, file=sys.stderr)
            if patience is not None and X_val is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping: best epoch {history['best_epoch']}", file=sys.stderr)
                break

        if best_weights is not None:
            for (param, _), best in zip(self.parameters(), best_weights):
                param[...] = best
        self.eval()
        return history

    # ----- inference -----------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> Tuple[float, float]:
        """Return (mean loss, accuracy) in inference mode. Accuracy compares argmax of outputs and targets."""
        self._check_xy(X, y)
        was_training = [layer.training for layer in self.layers]
        self.eval()
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), batch_size):
            outputs = self.forward(X[start:start + batch_size])
            targets = y[start:start + batch_size]
            total_loss += self.loss.forward(outputs, targets) * len(targets)
            correct += int(np.sum(np.argmax(outputs, axis=1) == np.argmax(targets, axis=1)))
        for layer, mode in zip(self.layers, was_training):
            layer.training = mode
        return total_loss / len(X), correct / len(X)

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Network outputs in inference mode, passed through the loss output transform
        (softmax probabilities for cross-entropy)."""
        self.eval()
        outputs = [self.forward(X[start:start + batch_size]) for start in range(0, len(X), batch_size)]
        return self.loss.output_transform(np.concatenate(outputs, axis=0))

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> List[str]:
        """Class name (from `classes`) of the most likely output for each sample."""
        probs = self.predict_proba(X, batch_size)
        if probs.shape[1] != len(self.classes):
            raise ValueError(f"Network has {probs.shape[1]} outputs but {len(self.classes)} classes.")
        return [self.classes[i] for i in np.argmax(probs, axis=1)]

    # ----- misc ----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "loss": self.loss.name,
            "optimizer": self.optimizer.get_config(),
            "layers": [layer.get_config() for layer in self.layers],
            "training": dict(self.training_config),
            "seed": self.seed,
        }

    @staticmethod
    def _check_xy(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Expected 2D X and y with the same non-zero length, got {X.shape} and {y.shape}.")
