"""--train and --predict pipelines of my_torch_analyzer.

Both modes load the network from LOADFILE, read CHESSFILE with the FEN
encoder, then either print one prediction per position (predict) or train the
network and save it (train). Every problem raises ValueError; main.py turns it
into a message on stderr and exit code 84.
"""
import sys
from typing import List, Optional, Sequence, TextIO

import numpy as np

from analyzer.cli import AnalyzerArgs
from analyzer.fen_encoder import INPUT_SIZE, encode_fens, encode_labels, read_chess_file
from mytorch.layers import Linear, Normalization
from mytorch.model import Sequential
from mytorch.serialization import load_network, read_metadata, save_network

# Positions are encoded and predicted by chunks, so a huge CHESSFILE never needs
# its whole (N, 773) float32 matrix in memory at once.
PREDICT_CHUNK = 1024


def network_input_size(model: Sequential) -> Optional[int]:
    """Number of features the first layer expects, or None if it cannot tell."""
    first = model.layers[0]
    if isinstance(first, Normalization):
        return first.features
    if isinstance(first, Linear):
        return first.in_features
    return None


def load_compatible_network(path: str) -> Sequential:
    """Load LOADFILE and check that it accepts the FEN encoder's input."""
    model = load_network(path)
    expected = network_input_size(model)
    if expected is not None and expected != INPUT_SIZE:
        raise ValueError(
            f"Network '{path}' expects {expected} input features, but the FEN encoder produces {INPUT_SIZE}."
        )
    return model


def predict(model: Sequential, fens: Sequence[str], out: TextIO) -> None:
    """Write one predicted label per position on `out`, in the input order."""
    for start in range(0, len(fens), PREDICT_CHUNK):
        labels = model.predict(encode_fens(fens[start:start + PREDICT_CHUNK]))
        out.write("".join(f"{label}\n" for label in labels))
    out.flush()


def run_predict(args: AnalyzerArgs, out: TextIO = sys.stdout) -> None:
    model = load_compatible_network(args.loadfile)
    # Expected outputs after the FEN are allowed in predict mode, and ignored.
    fens, _ = read_chess_file(args.chessfile, require_label=False)
    predict(model, fens, out)


def run_train(args: AnalyzerArgs, out: TextIO = sys.stdout) -> None:
    model = load_compatible_network(args.loadfile)
    fens, labels = read_chess_file(args.chessfile, require_label=True)
    X = encode_fens(fens)
    y = encode_labels(labels, model.classes)  # e.g. "Checkmate White" -> "Checkmate" for a 3-class network
    if not model.training_config:
        raise ValueError(f"Network '{args.loadfile}' has no training settings (epochs, batch_size).")

    # Same seed as the .conf: training the same network on the same file is reproducible.
    if model.seed is not None:
        np.random.seed(model.seed)
    history = model.fit(X, y, **model.training_config, verbose=True)  # per-epoch logs on stderr

    savefile = args.savefile or args.loadfile
    metadata = read_metadata(args.loadfile)
    metadata.update({"input_size": INPUT_SIZE, "trained_on": len(fens)})
    save_network(model, savefile, metadata=metadata)

    summary: List[str] = [f"Trained on {len(fens)} positions ({len(history['train_loss'])} epochs)"]
    if history["best_epoch"] is not None:
        best = history["best_epoch"] - 1
        summary.append(f"best epoch {history['best_epoch']}: val_loss {history['val_loss'][best]:.4f}, "
                       f"val_acc {history['val_acc'][best]:.4f}")
    else:
        summary.append(f"train_acc {history['train_acc'][-1]:.4f}")
    out.write(f"{' - '.join(summary)}\nNetwork saved to {savefile}\n")


def run(args: AnalyzerArgs) -> None:
    if args.mode == "predict":
        run_predict(args)
    else:
        run_train(args)
