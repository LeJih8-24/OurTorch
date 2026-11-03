"""Model selection: k-fold cross-validation.

A single train/validation split gives one score that depends on which samples
landed in the validation set. k-fold cross-validation splits the data into k
folds, trains k fresh networks (each one holding out a different fold) and
reports the mean and standard deviation of the k scores: a more reliable
estimate of how well a configuration generalizes, used to compare
hyperparameters [OPT1] [OPT2].
"""
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from mytorch.config import build_from_config
from mytorch.model import Sequential

Fold = Tuple[np.ndarray, np.ndarray]
ModelSource = Union[str, Dict[str, Any], Callable[[], Sequential]]


def _class_indices(y: np.ndarray) -> np.ndarray:
    """Class index of each sample: argmax of one-hot rows, or the values of a 1D array."""
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    if y.ndim == 1:
        return y
    raise ValueError(f"Labels must be one-hot (N, n_classes) or class indices (N,), got shape {y.shape}.")


def kfold_indices(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> List[Fold]:
    """Split range(n_samples) into k folds and return [(train_idx, val_idx), ...].

    Every sample is in exactly one validation fold; fold sizes differ by at
    most one. With `stratify` (one-hot targets or class indices), each class
    is split separately so every fold keeps the class proportions of the whole
    dataset: important when some classes are rare (checkmates).

    Shuffling uses np.random.default_rng(seed) when a seed is given, NumPy's
    global RNG otherwise (so np.random.seed() also makes it reproducible).
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k!r}.")
    if n_samples < k:
        raise ValueError(f"Cannot split {n_samples} samples into {k} folds.")
    rng = np.random.default_rng(seed) if seed is not None else np.random

    if stratify is None:
        groups = [np.arange(n_samples)]
    else:
        classes = _class_indices(stratify)
        if len(classes) != n_samples:
            raise ValueError(f"stratify has {len(classes)} labels for {n_samples} samples.")
        groups = [np.flatnonzero(classes == c) for c in np.unique(classes)]

    folds: List[List[np.ndarray]] = [[] for _ in range(k)]
    offset = 0
    for group in groups:
        if shuffle:
            group = rng.permutation(group)
        # Rotate the starting fold between classes so the extra samples of each
        # class do not all go to the first folds (keeps fold sizes balanced).
        for i, part in enumerate(np.array_split(group, k)):
            folds[(i + offset) % k].append(part)
        offset += len(group) % k

    result = []
    all_indices = np.arange(n_samples)
    for parts in folds:
        val_idx = np.sort(np.concatenate(parts))
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=True)
        result.append((train_idx, val_idx))
    return result


def _build(model_source: ModelSource) -> Sequential:
    if callable(model_source):
        model = model_source()
        if not isinstance(model, Sequential):
            raise ValueError("The model factory must return a Sequential.")
        return model
    return build_from_config(model_source)


def cross_validate(
    model_source: ModelSource,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    stratified: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    **fit_overrides: Any,
) -> Dict[str, Any]:
    """k-fold cross-validation of a network configuration.

    model_source: a .conf path, a config dict, or a function returning a new
    Sequential. A FRESH network is built for every fold, so nothing learned
    on one fold (weights, normalization statistics) leaks into another.

    For each fold, the network is trained on the k-1 other folds with its
    training_config (epochs, batch_size, val_split, patience), updated with
    `fit_overrides` (e.g. epochs=10). Early stopping uses the validation split
    taken INSIDE those training folds: the held-out fold is only used for the
    final score, so the score is not biased by early stopping.

    Returns a dict:
        fold_loss, fold_acc    one score per fold, on the held-out fold
        mean_loss, std_loss    mean and standard deviation over the folds
        mean_acc, std_acc
        best_epochs            epoch kept by early stopping in each fold
        histories              fit() history of each fold (learning curves)
        folds                  the (train_idx, val_idx) pairs used
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError(f"Expected 2D X and y with the same non-zero length, got {X.shape} and {y.shape}.")
    folds = kfold_indices(len(X), k, shuffle=True, stratify=y if stratified else None, seed=seed)

    fold_loss: List[float] = []
    fold_acc: List[float] = []
    histories: List[Dict[str, Any]] = []
    for number, (train_idx, val_idx) in enumerate(folds, start=1):
        model = _build(model_source)
        settings = {**model.training_config, **fit_overrides}
        missing = [key for key in ("epochs", "batch_size") if key not in settings]
        if missing:
            raise ValueError(f"Missing training setting(s) {missing}: pass them as keyword arguments.")
        history = model.fit(X[train_idx], y[train_idx], **settings)
        loss, accuracy = model.evaluate(X[val_idx], y[val_idx])
        fold_loss.append(loss)
        fold_acc.append(accuracy)
        histories.append(history)
        if verbose:
            # Logs go to stderr: stdout stays clean for the executables.
            print(f"fold {number}/{k} - loss {loss:.4f} - acc {accuracy:.4f} - best epoch {history['best_epoch']}",
                  file=sys.stderr)

    result = {
        "fold_loss": fold_loss,
        "fold_acc": fold_acc,
        "mean_loss": float(np.mean(fold_loss)),
        "std_loss": float(np.std(fold_loss)),
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "best_epochs": [history["best_epoch"] for history in histories],
        "histories": histories,
        "folds": folds,
    }
    if verbose:
        print(f"cross-validation - loss {result['mean_loss']:.4f} ± {result['std_loss']:.4f}"
              f" - acc {result['mean_acc']:.4f} ± {result['std_acc']:.4f}", file=sys.stderr)
    return result
