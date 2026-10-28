"""Network configuration files (.conf).

A .conf file is a JSON document describing a blank network:

    {
      "classes": ["Nothing", "Check", "Checkmate"],
      "loss": "cross_entropy",
      "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0001},
      "training": {"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5},
      "seed": 42,
      "layers": [
        {"type": "normalization"},
        {"type": "linear", "in": 773, "out": 256, "init": "he"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "in": 256, "out": 3, "init": "xavier"}
      ]
    }

validate_config() checks everything (types, ranges, unknown keys, layer
dimension continuity, output size == number of classes) and fills defaults.
Every problem raises ValueError with the path of the faulty field, so the
executables can report it on stderr and exit with 84.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from mytorch.initializers import get_initializer
from mytorch.layers import Dropout, LeakyReLU, Linear, Normalization, ReLU, Sigmoid, Softmax, Tanh
from mytorch.layers.base import Layer
from mytorch.losses import SoftmaxCrossEntropyLoss, get_loss
from mytorch.model import Sequential
from mytorch.optimizers import get_optimizer

TOP_LEVEL_DEFAULTS: Dict[str, Any] = {
    "loss": "cross_entropy",
    "optimizer": {"type": "adam"},
    "training": {},
    "seed": None,
}
REQUIRED_TOP_LEVEL = ("classes", "layers")

TRAINING_DEFAULTS: Dict[str, Any] = {"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5}

# type -> (default parameters, required parameters)
LAYER_SPECS: Dict[str, Dict[str, Any]] = {
    "linear": {"defaults": {"init": "he"}, "required": ("in", "out")},
    "relu": {"defaults": {}, "required": ()},
    "leakyrelu": {"defaults": {"alpha": 0.01}, "required": ()},
    "sigmoid": {"defaults": {}, "required": ()},
    "tanh": {"defaults": {}, "required": ()},
    "softmax": {"defaults": {}, "required": ()},
    "dropout": {"defaults": {"rate": 0.2, "mc_dropout": False}, "required": ()},
    "normalization": {"defaults": {"features": None}, "required": ()},
}


# ----- small type checkers -----------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_keys(section: Dict[str, Any], allowed: List[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    _require(not unknown, f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}.")


# ----- validation --------------------------------------------------------------

def _validate_layer(raw: Any, index: int) -> Dict[str, Any]:
    where = f"layers[{index}]"
    _require(isinstance(raw, dict), f"{where} must be an object, got {raw!r}.")
    _require("type" in raw, f"{where} is missing 'type'.")
    layer_type = str(raw["type"]).strip().lower()
    _require(layer_type in LAYER_SPECS, f"{where}: unknown layer type '{raw['type']}'. Available: {sorted(LAYER_SPECS)}.")
    spec = LAYER_SPECS[layer_type]
    _check_keys(raw, ["type", *spec["required"], *spec["defaults"]], where)
    for key in spec["required"]:
        _require(key in raw, f"{where} ({layer_type}) is missing '{key}'.")

    layer = {"type": layer_type, **spec["defaults"], **{k: v for k, v in raw.items() if k != "type"}}
    if layer_type == "linear":
        for key in ("in", "out"):
            _require(_is_int(layer[key]) and layer[key] > 0, f"{where}.{key} must be a positive integer, got {layer[key]!r}.")
        try:
            get_initializer(layer["init"])
        except ValueError as exc:
            raise ValueError(f"{where}.init: {exc}") from exc
    elif layer_type == "leakyrelu":
        _require(_is_number(layer["alpha"]) and layer["alpha"] >= 0, f"{where}.alpha must be a number >= 0.")
    elif layer_type == "dropout":
        _require(_is_number(layer["rate"]) and 0 <= layer["rate"] < 1, f"{where}.rate must be in [0, 1), got {layer['rate']!r}.")
        _require(isinstance(layer["mc_dropout"], bool), f"{where}.mc_dropout must be true or false.")
    elif layer_type == "normalization":
        features = layer["features"]
        _require(features is None or (_is_int(features) and features > 0),
                 f"{where}.features must be a positive integer, got {features!r}.")
    return layer


def _check_dimensions(layers: List[Dict[str, Any]], n_classes: int) -> None:
    """Infer missing Normalization sizes and check that consecutive Linear sizes match."""
    linears = [layer for layer in layers if layer["type"] == "linear"]
    _require(bool(linears), "layers must contain at least one 'linear' layer.")
    width: Optional[int] = None
    for index, layer in enumerate(layers):
        if layer["type"] == "normalization":
            _require(index == 0, f"layers[{index}]: 'normalization' is only allowed as the first layer.")
            if layer["features"] is None:
                layer["features"] = linears[0]["in"]
            width = layer["features"]
        elif layer["type"] == "linear":
            _require(width is None or layer["in"] == width,
                     f"layers[{index}].in is {layer['in']} but the previous layer outputs {width} features.")
            width = layer["out"]
    _require(width == n_classes,
             f"The network outputs {width} values but 'classes' has {n_classes} entries.")


def _validate_training(raw: Any) -> Dict[str, Any]:
    _require(isinstance(raw, dict), "training must be an object.")
    _check_keys(raw, list(TRAINING_DEFAULTS), "training")
    training = {**TRAINING_DEFAULTS, **raw}
    for key in ("epochs", "batch_size"):
        _require(_is_int(training[key]) and training[key] > 0, f"training.{key} must be a positive integer, got {training[key]!r}.")
    _require(_is_number(training["val_split"]) and 0 <= training["val_split"] < 1,
             f"training.val_split must be in [0, 1), got {training['val_split']!r}.")
    patience = training["patience"]
    _require(patience is None or (_is_int(patience) and patience > 0),
             f"training.patience must be a positive integer or null, got {patience!r}.")
    return training


def validate_config(raw: Any) -> Dict[str, Any]:
    """Return a complete, validated copy of a configuration dict (defaults filled in)."""
    _require(isinstance(raw, dict), "The configuration must be a JSON object.")
    _check_keys(raw, [*REQUIRED_TOP_LEVEL, *TOP_LEVEL_DEFAULTS], "config")
    for key in REQUIRED_TOP_LEVEL:
        _require(key in raw, f"config is missing '{key}'.")
    config = {**TOP_LEVEL_DEFAULTS, **raw}

    classes = config["classes"]
    _require(isinstance(classes, list) and len(classes) >= 2, "classes must be a list of at least 2 labels.")
    _require(all(isinstance(c, str) and c.strip() for c in classes), "classes must only contain non-empty strings.")
    _require(len(set(classes)) == len(classes), "classes must not contain duplicates.")

    try:
        loss = get_loss(config["loss"])
        optimizer_config = dict(config["optimizer"]) if isinstance(config["optimizer"], dict) else config["optimizer"]
        get_optimizer(optimizer_config)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid loss/optimizer: {exc}") from exc

    _require(isinstance(config["layers"], list) and config["layers"], "layers must be a non-empty list.")
    layers = [_validate_layer(layer, i) for i, layer in enumerate(config["layers"])]
    _check_dimensions(layers, len(classes))
    _require(not (isinstance(loss, SoftmaxCrossEntropyLoss) and layers[-1]["type"] == "softmax"),
             "Do not end the network with 'softmax' when the loss is 'cross_entropy': the loss already applies it.")

    seed = config["seed"]
    _require(seed is None or (_is_int(seed) and seed >= 0), f"seed must be a non-negative integer or null, got {seed!r}.")

    return {
        "classes": list(classes),
        "loss": loss.name,
        "optimizer": optimizer_config,
        "training": _validate_training(config["training"]),
        "seed": seed,
        "layers": layers,
    }


# ----- building ------------------------------------------------------------------

_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Layer]] = {
    "linear": lambda c: Linear(c["in"], c["out"], init=c["init"]),
    "relu": lambda c: ReLU(),
    "leakyrelu": lambda c: LeakyReLU(c["alpha"]),
    "sigmoid": lambda c: Sigmoid(),
    "tanh": lambda c: Tanh(),
    "softmax": lambda c: Softmax(),
    "dropout": lambda c: Dropout(c["rate"], mc_dropout=c["mc_dropout"]),
    "normalization": lambda c: Normalization(c["features"]),
}


def load_config(path: str) -> Dict[str, Any]:
    """Read and validate a .conf file. Raises ValueError (unreadable file, bad JSON, invalid config)."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file '{path}': {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration file '{path}' is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    try:
        return validate_config(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid configuration file '{path}': {exc}") from exc


def build_from_config(config: Union[str, Dict[str, Any]]) -> Sequential:
    """Build a freshly initialized Sequential from a .conf path or a config dict.

    If the config has a seed, np.random.seed(seed) is called first, so the
    initial weights (and a training run started right after) are reproducible.
    """
    config = load_config(config) if isinstance(config, str) else validate_config(config)
    if config["seed"] is not None:
        np.random.seed(config["seed"])
    layers = [_BUILDERS[layer["type"]](layer) for layer in config["layers"]]
    return Sequential(
        layers,
        get_loss(config["loss"]),
        get_optimizer(dict(config["optimizer"])),
        config["classes"],
        training_config=config["training"],
        seed=config["seed"],
    )
