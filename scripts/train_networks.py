#!/usr/bin/env python3
"""Train the pre-trained networks pushed with the project (my_torch_network_*.nn).

Usage (from the repository root, with NumPy available):
    python3 scripts/train_networks.py [--levels basic medium full] [--per-file N]
                                      [--seed 42] [--epochs E] [--out DIR] [--no-save]

For every level, the network is built from configs/<level>_network.conf and
trained on the dataset/ files with the same split for all levels:
    80 % training, 10 % validation (early stopping), 10 % test (final score),
stratified on the 5 dataset labels and fixed by --seed. The test positions are
never used for training or for early stopping.

Outputs:
    my_torch_network_<level>.nn at the repository root (unless --no-save)
    <out>/<level>_report.txt    accuracy, per-class metrics, confusion matrix on the test set
    <out>/<level>_history.json  learning curves (train / validation loss and accuracy)
    <out>/summary.json          one line of results per level

Running it twice with the same arguments produces the same networks.
"""
import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from analyzer.fen_encoder import INPUT_SIZE, adapt_label, encode_fens, encode_labels, read_chess_file  # noqa: E402
from evaluate import evaluate  # noqa: E402
from mytorch.config import build_from_config  # noqa: E402
from mytorch.serialization import save_network  # noqa: E402

LEVELS = ("basic", "medium", "full")
ENCODE_CHUNK = 65536


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_dataset(per_file: int) -> Tuple[List[str], List[str], List[str]]:
    """All positions and labels of dataset/*/*.txt (at most `per_file` lines per file if > 0)."""
    files = sorted(glob.glob(os.path.join(ROOT, "dataset", "*", "*.txt")))
    if not files:
        raise ValueError(f"No dataset file found in {os.path.join(ROOT, 'dataset')}.")
    fens: List[str] = []
    labels: List[str] = []
    for path in files:
        file_fens, file_labels = read_chess_file(path, require_label=True)
        if per_file > 0:
            file_fens, file_labels = file_fens[:per_file], file_labels[:per_file]
        fens += file_fens
        labels += file_labels
    return fens, labels, [os.path.relpath(path, ROOT) for path in files]


def encode_inputs(fens: List[str]) -> np.ndarray:
    """(N, INPUT_SIZE) matrix stored as uint8: the encoding only contains 0 and 1,
    so this takes 4 times less memory than float32 (560 MB for the whole dataset)."""
    X = np.zeros((len(fens), INPUT_SIZE), dtype=np.uint8)
    for start in range(0, len(fens), ENCODE_CHUNK):
        X[start:start + ENCODE_CHUNK] = encode_fens(fens[start:start + ENCODE_CHUNK])
    return X


def stratified_split(labels: List[str], seed: int, val: float = 0.1, test: float = 0.1) -> Dict[str, np.ndarray]:
    """Indices of the train / val / test parts, with the same label proportions in each."""
    rng = np.random.default_rng(seed)
    labels_array = np.array(labels)
    parts: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for label in sorted(set(labels)):
        indices = rng.permutation(np.flatnonzero(labels_array == label))
        n_test, n_val = int(round(len(indices) * test)), int(round(len(indices) * val))
        parts["test"].append(indices[:n_test])
        parts["val"].append(indices[n_test:n_test + n_val])
        parts["train"].append(indices[n_test + n_val:])
    return {name: np.sort(np.concatenate(chunks)) for name, chunks in parts.items()}


def train_level(level: str, X: np.ndarray, labels: List[str], split: Dict[str, np.ndarray],
                args: argparse.Namespace) -> Dict[str, object]:
    model = build_from_config(os.path.join(ROOT, "configs", f"{level}_network.conf"))
    y = encode_labels(labels, model.classes)
    settings = dict(model.training_config)
    settings.pop("val_split", None)  # the fixed validation part is used instead
    if args.epochs:
        settings["epochs"] = args.epochs

    log(f"\n=== {level}: {model.classes}, {sum(p.size for p, _ in model.parameters())} parameters, "
        f"{len(split['train'])} training positions")
    if model.seed is not None:
        np.random.seed(model.seed)
    start = time.time()
    history = model.fit(X[split["train"]], y[split["train"]], **settings,
                        validation_data=(X[split["val"]], y[split["val"]]), verbose=True)
    duration = time.time() - start

    test_idx = split["test"]
    predictions = model.predict(X[test_idx])
    expected = [adapt_label(labels[i], model.classes) for i in test_idx]
    accuracy = float(np.mean([p == e for p, e in zip(predictions, expected)]))
    report = evaluate(model.classes, predictions, expected)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{level}_report.txt"), "w", encoding="utf-8") as file:
        file.write(report + "\n")
    with open(os.path.join(args.out, f"{level}_history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=1)

    result = {
        "level": level,
        "classes": model.classes,
        "test_accuracy": accuracy,
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "train_positions": int(len(split["train"])),
        "val_positions": int(len(split["val"])),
        "test_positions": int(len(test_idx)),
        "training_seconds": round(duration, 1),
    }
    if not args.no_save:
        path = os.path.join(ROOT, f"my_torch_network_{level}.nn")
        save_network(model, path, metadata={
            "input_size": INPUT_SIZE,
            "trained_on": int(len(split["train"])),
            "test_accuracy": round(accuracy, 4),
            "training_script": "scripts/train_networks.py",
            "split_seed": args.seed,
            "per_file": args.per_file,
        })
        result["network"] = os.path.relpath(path, ROOT)
    log(f"{level}: test accuracy {accuracy:.4f} (best epoch {history['best_epoch']}, {duration:.0f}s)")
    log(report)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the pre-trained my_torch_network_*.nn networks.")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--per-file", type=int, default=0, help="lines per dataset file (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the train / val / test split")
    parser.add_argument("--epochs", type=int, default=0, help="override the maximum number of epochs")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "output", "train_networks"))
    parser.add_argument("--no-save", action="store_true", help="do not write my_torch_network_*.nn")
    args = parser.parse_args()

    try:
        start = time.time()
        fens, labels, files = load_dataset(args.per_file)
        X = encode_inputs(fens)
        split = stratified_split(labels, args.seed)
        log(f"{len(fens)} positions from {len(files)} files encoded in {time.time() - start:.0f}s "
            f"(train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])})")
        results = [train_level(level, X, labels, split, args) for level in args.levels]
    except (ValueError, OSError, MemoryError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 84

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as file:
        json.dump({"dataset_files": files, "positions": len(fens), "split_seed": args.seed,
                   "per_file": args.per_file, "results": results}, file, indent=2)
    for result in results:
        print(f"{result['level']:>6}: test accuracy {result['test_accuracy']:.4f} "
              f"- best epoch {result['best_epoch']}/{result['epochs_run']} - {result['training_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
