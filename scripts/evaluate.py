#!/usr/bin/env python3
"""Compare my_torch_analyzer predictions with the expected labels of a CHESSFILE.

Usage:
    python3 scripts/evaluate.py NETWORK PREDICTIONS CHESSFILE

NETWORK      the .nn used for the predictions (gives the classes)
PREDICTIONS  output of `my_torch_analyzer --predict NETWORK CHESSFILE`
CHESSFILE    the same positions, each followed by its expected label

Expected labels are adapted to the network classes exactly as in --train
("Checkmate White" -> "Checkmate" for a 3-class network). Prints the accuracy,
precision / recall / F1 per class and the confusion matrix.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer.fen_encoder import adapt_label, read_chess_file  # noqa: E402
from mytorch.serialization import load_network  # noqa: E402


def evaluate(classes, predictions, expected) -> str:
    index = {name: i for i, name in enumerate(classes)}
    k = len(classes)
    matrix = [[0] * k for _ in range(k)]
    for pred, true in zip(predictions, expected):
        if pred not in index:
            raise ValueError(f"Prediction {pred!r} is not a class of the network {classes}.")
        matrix[index[true]][index[pred]] += 1

    total = len(expected)
    correct = sum(matrix[i][i] for i in range(k))
    width = max(len(name) for name in classes)
    lines = [f"accuracy {correct / total:.4f} ({correct}/{total})", "",
             f"{'class':<{width}}  precision  recall     F1  support"]
    for i, name in enumerate(classes):
        predicted = sum(matrix[r][i] for r in range(k))
        support = sum(matrix[i])
        precision = matrix[i][i] / predicted if predicted else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"{name:<{width}}  {precision:9.3f}  {recall:6.3f}  {f1:5.3f}  {support:7d}")
    lines += ["", "confusion matrix (rows = expected, columns = predicted)",
              " " * (width + 2) + "  ".join(f"{c[:10]:>10}" for c in classes)]
    for i, name in enumerate(classes):
        lines.append(f"{name:<{width}}  " + "  ".join(f"{n:>10d}" for n in matrix[i]))
    return "\n".join(lines)


def main(argv) -> int:
    if len(argv) != 3:
        sys.stderr.write("Error: usage: python3 scripts/evaluate.py NETWORK PREDICTIONS CHESSFILE\n")
        return 84
    network, predictions_path, chessfile = argv
    try:
        classes = load_network(network).classes
        with open(predictions_path, encoding="utf-8") as file:
            predictions = [line.strip() for line in file if line.strip()]
        _, labels = read_chess_file(chessfile, require_label=True)
        if len(predictions) != len(labels):
            raise ValueError(f"{len(predictions)} predictions for {len(labels)} positions.")
        print(evaluate(classes, predictions, [adapt_label(label, classes) for label in labels]))
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
