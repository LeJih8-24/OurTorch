"""FEN parsing and encoding of chess positions into network inputs and targets.

Label convention (from the subject's example): the color is the side that
*gives* check or checkmate. A position where White is to move and is mated is
labelled "Checkmate Black".
"""
import re
from typing import List, Optional, Sequence, Tuple
import numpy as np

PIECES = "PNBRQKpnbrqk"
CASTLING_RIGHTS = "KQkq"
BOARD_FEATURES = len(PIECES) * 64
INPUT_SIZE = BOARD_FEATURES + 1 + len(CASTLING_RIGHTS)

FULL_CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
KNOWN_LABELS = FULL_CLASSES + ["Check", "Checkmate", "Stalemate"]

_CASTLING_RE = re.compile(r"^(-|K?Q?k?q?)$")
_EN_PASSANT_RE = re.compile(r"^(-|[a-h][36])$")


def _validate_board(board: str) -> None:
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN board must have 8 ranks separated by '/', got {len(ranks)}.")
    for index, rank in enumerate(ranks):
        squares = 0
        for char in rank:
            if char in "12345678":
                squares += int(char)
            elif char in PIECES:
                squares += 1
            else:
                raise ValueError(f"Invalid character {char!r} in FEN rank {8 - index}.")
        if squares != 8:
            raise ValueError(f"FEN rank {8 - index} describes {squares} squares instead of 8.")


def validate_fen(fen: str) -> List[str]:
    """Check a 4- or 6-field FEN string and return its fields. Raises ValueError if invalid."""
    fields = fen.split()
    if len(fields) not in (4, 6):
        raise ValueError(f"FEN must have 4 or 6 fields, got {len(fields)}: {fen!r}.")
    _validate_board(fields[0])
    if fields[1] not in ("w", "b"):
        raise ValueError(f"FEN active color must be 'w' or 'b', got {fields[1]!r}.")
    if fields[2] == "" or not _CASTLING_RE.match(fields[2]):
        raise ValueError(f"Invalid FEN castling field {fields[2]!r}.")
    if not _EN_PASSANT_RE.match(fields[3]):
        raise ValueError(f"Invalid FEN en passant field {fields[3]!r}.")
    for name, value in zip(("halfmove clock", "fullmove number"), fields[4:]):
        if not value.isdigit():
            raise ValueError(f"FEN {name} must be a non-negative integer, got {value!r}.")
    return fields


def parse_chess_line(line: str, require_label: bool) -> Tuple[str, Optional[str]]:
    """Split a CHESSFILE line into (fen, label). The label is None when absent.

    The FEN has 6 fields, or 4 when the move counters are omitted. Everything
    after it is the label (e.g. "Checkmate Black").
    """
    tokens = line.split()
    if len(tokens) < 4:
        raise ValueError(f"Line is too short to contain a FEN: {line.strip()!r}.")
    has_counters = len(tokens) >= 6 and tokens[4].isdigit() and tokens[5].isdigit()
    n_fields = 6 if has_counters else 4
    fen = " ".join(tokens[:n_fields])
    validate_fen(fen)
    label = " ".join(tokens[n_fields:]) or None
    if require_label:
        if label is None:
            raise ValueError(f"Missing expected output after FEN {fen!r}.")
        if label not in KNOWN_LABELS:
            raise ValueError(f"Unknown label {label!r}. Expected one of {KNOWN_LABELS}.")
    return fen, label


def read_chess_file(path: str, require_label: bool) -> Tuple[List[str], List[Optional[str]]]:
    """Parse every non-empty line of `path`. Errors mention the faulty line number."""
    fens: List[str] = []
    labels: List[Optional[str]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                fen, label = parse_chess_line(line, require_label)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            fens.append(fen)
            labels.append(label)
    if not fens:
        raise ValueError(f"{path} contains no chessboard.")
    return fens, labels


def encode_fen(fen: str) -> np.ndarray:
    """Encode a FEN into a float32 vector of size INPUT_SIZE (773).

    - [0, 768): one-hot planes, index = piece * 64 + square, with pieces in
      PIECES order and square 0 = a8, 7 = h8, ..., 63 = h1 (FEN reading order).
    - 768: 1.0 if White is to move, else 0.0.
    - 769..772: castling rights K, Q, k, q.
    """
    fields = validate_fen(fen)
    vector = np.zeros(INPUT_SIZE, dtype=np.float32)
    square = 0
    for char in fields[0]:
        if char == "/":
            continue
        if char.isdigit():
            square += int(char)
        else:
            vector[PIECES.index(char) * 64 + square] = 1.0
            square += 1
    vector[BOARD_FEATURES] = 1.0 if fields[1] == "w" else 0.0
    for offset, right in enumerate(CASTLING_RIGHTS):
        if right in fields[2]:
            vector[BOARD_FEATURES + 1 + offset] = 1.0
    return vector


def encode_fens(fens: Sequence[str]) -> np.ndarray:
    """Encode several FENs into a (N, INPUT_SIZE) float32 matrix."""
    matrix = np.zeros((len(fens), INPUT_SIZE), dtype=np.float32)
    for row, fen in enumerate(fens):
        matrix[row] = encode_fen(fen)
    return matrix


def adapt_label(label: str, classes: Sequence[str]) -> str:
    """Map a dataset label onto the classes of the loaded network.

    Finer labels are coarsened when the network does not distinguish them:
    "Checkmate White" -> "Checkmate" -> "Check", and "Stalemate" -> "Nothing".
    """
    candidates = [label]
    words = label.split()
    if len(words) == 2:
        candidates.append(words[0])
    if words[0] == "Checkmate":
        candidates.append("Check")
    if label == "Stalemate":
        candidates.append("Nothing")
    for candidate in candidates:
        if candidate in classes:
            return candidate
    raise ValueError(f"Label {label!r} cannot be mapped to network classes {list(classes)}.")


def encode_labels(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """One-hot encode labels into a (N, len(classes)) float32 matrix, in `classes` order."""
    index = {name: i for i, name in enumerate(classes)}
    targets = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        targets[row, index[adapt_label(label, classes)]] = 1.0
    return targets
