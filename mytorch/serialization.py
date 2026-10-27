"""Network files (.nn): save and load a complete network.

A .nn file is a NumPy .npz archive (a zip of .npy arrays), loaded with
allow_pickle=False so a file can never execute code. It contains:

    header          JSON string: {"format": "mytorch-nn", "version": 1, "metadata": {...}}
    config          JSON string: model.get_config() (classes, loss, optimizer, training, seed, layers)
    layer{i}.{key}  one array per layer state entry (Linear: W, b; Normalization: mean, std, count)

Loading rebuilds the architecture from `config` with build_from_config(), then
copies every saved array into the layers with load_state_dict(), which checks
names and shapes. Any problem raises ValueError, so the executables can print
it on stderr and exit with 84.

The optimizer state (momentum, Adam moments) is NOT saved: retraining a
loaded network starts with a fresh optimizer, configured from `config`.
"""
import json
import os
import tempfile
import zipfile
from typing import Any, Dict, Optional
import numpy as np

from mytorch.config import build_from_config
from mytorch.model import Sequential

FORMAT_NAME = "mytorch-nn"
FORMAT_VERSION = 1


def _state_name(index: int, key: str) -> str:
    return f"layer{index}.{key}"


def save_network(model: Sequential, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save `model` to `path` (written atomically: an existing file is only replaced once complete).

    `metadata` is any JSON-serializable dict stored in the header (e.g. the FEN
    encoder version), returned by read_metadata().
    """
    header = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "metadata": metadata or {}}
    arrays: Dict[str, np.ndarray] = {
        "header": np.array(json.dumps(header)),
        "config": np.array(json.dumps(model.get_config())),
    }
    for index, layer in enumerate(model.layers):
        for key, value in layer.state_dict().items():
            arrays[_state_name(index, key)] = np.asarray(value)

    directory = os.path.dirname(os.path.abspath(path))
    try:
        # Write to a temporary file in the same directory, then rename: if anything
        # fails, the previous network file (e.g. LOADFILE in --train) is left intact.
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".nn")
        try:
            with os.fdopen(fd, "wb") as file:
                np.savez_compressed(file, **arrays)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except OSError as exc:
        raise ValueError(f"Cannot write network file '{path}': {exc.strerror or exc}") from exc


def _read_archive(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        raise ValueError(f"Cannot read network file '{path}': no such readable file.")
    try:
        # zipfile check first: np.load() also accepts single .npy files and text-like data
        if not zipfile.is_zipfile(path):
            raise ValueError("not an .npz archive")
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except OSError as exc:
        raise ValueError(f"Cannot read network file '{path}': {exc.strerror or exc}") from exc
    except (ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"'{path}' is not a valid network file (corrupted or wrong format): {exc}") from exc


def _read_json(arrays: Dict[str, np.ndarray], name: str, path: str) -> Any:
    if name not in arrays:
        raise ValueError(f"Network file '{path}' is missing its '{name}' entry.")
    try:
        return json.loads(str(arrays[name]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Network file '{path}' has a corrupted '{name}' entry.") from exc


def _read_header(arrays: Dict[str, np.ndarray], path: str) -> Dict[str, Any]:
    header = _read_json(arrays, "header", path)
    if not isinstance(header, dict) or header.get("format") != FORMAT_NAME:
        raise ValueError(f"'{path}' is not a {FORMAT_NAME} network file.")
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"Network file '{path}' uses format version {header.get('version')}, "
            f"this program reads version {FORMAT_VERSION}."
        )
    return header


def load_network(path: str) -> Sequential:
    """Load a network saved by save_network(). Raises ValueError on any problem."""
    arrays = _read_archive(path)
    _read_header(arrays, path)
    try:
        model = build_from_config(_read_json(arrays, "config", path))
    except ValueError as exc:
        raise ValueError(f"Network file '{path}' has an invalid configuration: {exc}") from exc

    expected = set()
    for index, layer in enumerate(model.layers):
        state = {}
        for key in layer.state_keys:
            name = _state_name(index, key)
            expected.add(name)
            if name not in arrays:
                raise ValueError(f"Network file '{path}' is missing the array '{name}'.")
            if not np.issubdtype(arrays[name].dtype, np.number):
                raise ValueError(f"Network file '{path}': array '{name}' is not numeric.")
            state[key] = arrays[name]
        try:
            layer.load_state_dict(state)
        except ValueError as exc:
            raise ValueError(f"Network file '{path}', layer {index}: {exc}") from exc

    unexpected = sorted(set(arrays) - expected - {"header", "config"})
    if unexpected:
        raise ValueError(f"Network file '{path}' contains unexpected arrays: {unexpected}.")
    return model


def read_metadata(path: str) -> Dict[str, Any]:
    """Return the metadata dict stored by save_network(), without building the network."""
    return _read_header(_read_archive(path), path).get("metadata", {})
