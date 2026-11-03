"""Model selection: k-fold cross-validation and grid search.

A single train/validation split gives one score that depends on which samples
landed in the validation set. k-fold cross-validation splits the data into k
folds, trains k fresh networks (each one holding out a different fold) and
reports the mean and standard deviation of the k scores: a more reliable
estimate of how well a configuration generalizes, used to compare
hyperparameters [OPT1] [OPT2].

Grid search tries every combination of a grid of hyperparameter values on the
same data and ranks them [OPT2].
"""
import copy
import itertools
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from mytorch.config import build_from_config, load_config, validate_config
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


# ----- grid search -----------------------------------------------------------

ParamGrid = Dict[str, Sequence[Any]]

METRICS = {"loss": "minimize", "accuracy": "maximize"}


def parameter_combinations(param_grid: ParamGrid) -> List[Dict[str, Any]]:
    """Every combination of the grid, as a list of {path: value} dicts (cartesian product).

    {"optimizer.lr": [0.001, 0.01], "layers.*.rate": [0.1, 0.3]} gives 4 combinations.
    """
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict {path: [values, ...]}.")
    for path, values in param_grid.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"param_grid keys must be non-empty strings, got {path!r}.")
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"param_grid['{path}'] must be a non-empty list of values, got {values!r}.")
    paths = list(param_grid)
    return [dict(zip(paths, values)) for values in itertools.product(*(param_grid[p] for p in paths))]


def _assign(node: Any, keys: List[str], value: Any, path: str, wildcard: bool) -> int:
    """Set `value` at `keys` inside `node`; return how many places were changed.

    Under a "*" (wildcard), elements that do not have the requested key are
    skipped instead of raising: "layers.*.rate" only changes dropout layers.
    """
    key, rest = keys[0], keys[1:]
    if isinstance(node, list):
        if key == "*":
            if not rest:
                raise ValueError(f"'{path}': '*' must be followed by a key (e.g. 'layers.*.rate').")
            return sum(_assign(item, rest, value, path, True) for item in node)
        if not key.isdigit() or int(key) >= len(node):
            raise ValueError(f"'{path}': '{key}' is not a valid index (the list has {len(node)} elements).")
        if not rest:
            node[int(key)] = copy.deepcopy(value)
            return 1
        return _assign(node[int(key)], rest, value, path, wildcard)
    if isinstance(node, dict):
        if not rest:
            if wildcard and key not in node:
                return 0
            node[key] = copy.deepcopy(value)
            return 1
        if key not in node:
            if wildcard:
                return 0
            raise ValueError(f"'{path}': no key '{key}' in the configuration.")
        return _assign(node[key], rest, value, path, wildcard)
    raise ValueError(f"'{path}': cannot go into a {type(node).__name__} with '{key}'.")


def set_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a config dict, in place, with a dotted path.

    "optimizer.lr"        -> config["optimizer"]["lr"]
    "training.batch_size" -> config["training"]["batch_size"]
    "layers.1.out"        -> config["layers"][1]["out"]
    "layers.*.rate"       -> "rate" of every layer that has one (all dropouts)
    "layers"              -> the whole list of layers (to compare architectures)
    """
    if _assign(config, path.split("."), value, path, False) == 0:
        raise ValueError(f"'{path}' matches nothing in the configuration.")


def _holdout_indices(y: np.ndarray, val_fraction: float, stratified: bool, seed: Optional[int]) -> Fold:
    """One train/validation split, the same for every combination (fair comparison)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_split must be in (0, 1) for grid search, got {val_fraction}.")
    rng = np.random.default_rng(seed) if seed is not None else np.random
    groups = [np.flatnonzero(_class_indices(y) == c) for c in np.unique(_class_indices(y))] if stratified \
        else [np.arange(len(y))]
    val_parts = [rng.permutation(group)[:int(round(len(group) * val_fraction))] for group in groups]
    val_idx = np.sort(np.concatenate(val_parts)).astype(int)
    if len(val_idx) == 0 or len(val_idx) == len(y):
        raise ValueError(f"Cannot hold out {val_fraction:.0%} of {len(y)} samples for validation.")
    return np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=True), val_idx


def _describe(params: Dict[str, Any]) -> str:
    return " ".join(f"{path}={value!r}" if not isinstance(value, list) else f"{path}=<{len(value)} items>"
                    for path, value in params.items())


def grid_search(
    base_config: Union[str, Dict[str, Any]],
    param_grid: ParamGrid,
    X: np.ndarray,
    y: np.ndarray,
    k: Optional[int] = None,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    val_split: Optional[float] = None,
    metric: str = "loss",
    stratified: bool = True,
    seed: Optional[int] = None,
    refit: bool = False,
    verbose: bool = False,
    **fit_overrides: Any,
) -> Dict[str, Any]:
    """Try every combination of `param_grid` on top of `base_config` and rank them [OPT2].

    base_config: a .conf path or a config dict. param_grid: {path: [values]},
    with the dotted paths of set_config_value(), e.g.
        {"optimizer.lr": [0.0003, 0.001, 0.003], "layers.*.rate": [0.1, 0.3]}

    Every combination is validated BEFORE any training: a wrong path or an
    invalid value stops the search immediately, with the faulty combination.

    How each combination is scored (always on data it was not trained on):
    - k given: k-fold cross-validation (cross_validate), score = mean over folds;
    - validation_data given: trained on (X, y), scored on validation_data;
    - otherwise: one stratified hold-out split of (X, y), `val_split` (default:
      the config's training.val_split), identical for every combination.
    With a hold-out set, it also drives early stopping, so its score is slightly
    optimistic: keep a separate test set for the final measure.

    metric: "loss" (lower is better) or "accuracy" (higher is better).
    fit_overrides replace training settings for every combination (e.g. epochs=10).
    refit=True trains the best configuration once more on all of (X, y) and
    returns it as best_model.

    Returns a dict:
        results          one entry per combination, best first: params, score,
                         loss, accuracy, loss_std / accuracy_std (k-fold only),
                         best_epochs, config
        best_params, best_score, best_config (ready for build_from_config or a .conf file)
        metric, n_combinations, best_model (only with refit=True)
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}.")
    if k is not None and validation_data is not None:
        raise ValueError("Use either k (cross-validation) or validation_data, not both.")
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError(f"Expected 2D X and y with the same non-zero length, got {X.shape} and {y.shape}.")

    base = load_config(base_config) if isinstance(base_config, str) else validate_config(base_config)
    combinations = parameter_combinations(param_grid)
    configs = []
    for params in combinations:
        config = copy.deepcopy(base)
        try:
            for path, value in params.items():
                set_config_value(config, path, value)
            configs.append(validate_config(config))
        except ValueError as exc:
            raise ValueError(f"Invalid combination {_describe(params)}: {exc}") from exc

    if k is None:
        if validation_data is not None:
            X_train, y_train = X, y
            X_val, y_val = (np.asarray(a) for a in validation_data)
        else:
            fraction = base["training"]["val_split"] if val_split is None else val_split
            train_idx, val_idx = _holdout_indices(y, fraction or 0.15, stratified, seed)
            X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    results = []
    for number, (params, config) in enumerate(zip(combinations, configs), start=1):
        if k is not None:
            cv = cross_validate(config, X, y, k=k, stratified=stratified, seed=seed, **fit_overrides)
            entry = {"loss": cv["mean_loss"], "accuracy": cv["mean_acc"],
                     "loss_std": cv["std_loss"], "accuracy_std": cv["std_acc"], "best_epochs": cv["best_epochs"]}
        else:
            model = build_from_config(config)
            settings = {**model.training_config, **fit_overrides, "validation_data": (X_val, y_val)}
            history = model.fit(X_train, y_train, **settings)
            loss, accuracy = model.evaluate(X_val, y_val)
            entry = {"loss": loss, "accuracy": accuracy, "loss_std": None, "accuracy_std": None,
                     "best_epochs": [history["best_epoch"]]}
        entry = {"params": params, "score": entry[metric], **entry, "config": config}
        results.append(entry)
        if verbose:
            # Logs go to stderr: stdout stays clean for the executables.
            print(f"[{number}/{len(combinations)}] {_describe(params)} -> loss {entry['loss']:.4f}"
                  f" - acc {entry['accuracy']:.4f}", file=sys.stderr)

    results.sort(key=lambda r: r["score"], reverse=METRICS[metric] == "maximize")
    best = results[0]
    output = {
        "results": results,
        "best_params": best["params"],
        "best_score": best["score"],
        "best_config": copy.deepcopy(best["config"]),
        "metric": metric,
        "n_combinations": len(results),
    }
    if verbose:
        print(f"best ({metric} {best['score']:.4f}): {_describe(best['params'])}", file=sys.stderr)
    if refit:
        model = build_from_config(best["config"])
        settings = {**model.training_config, **fit_overrides}
        if validation_data is not None:
            settings["validation_data"] = (X_val, y_val)
        model.fit(X, y, **settings)
        output["best_model"] = model
    return output
