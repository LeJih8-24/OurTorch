from pathlib import Path
import tempfile
import unittest
import numpy as np

from analyzer.fen_encoder import (
    BOARD_FEATURES,
    FULL_CLASSES,
    INPUT_SIZE,
    PIECES,
    adapt_label,
    encode_fen,
    encode_fens,
    encode_labels,
    parse_chess_line,
    read_chess_file,
    validate_fen,
)

# The 5 boards from page 6 of the subject.
SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
SUBJECT_FULL_LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]
START_FEN = SUBJECT_FENS[1]


def feature_index(piece: str, square: str) -> int:
    """Index of `piece` on `square` (e.g. "e2") in the encoded vector."""
    file = "abcdefgh".index(square[0])
    rank = int(square[1])
    return PIECES.index(piece) * 64 + (8 - rank) * 8 + file


class TestParsing(unittest.TestCase):
    def test_subject_boards_are_valid(self):
        for fen in SUBJECT_FENS:
            self.assertEqual(len(validate_fen(fen)), 6)
            self.assertEqual(parse_chess_line(fen, require_label=False), (fen, None))

    def test_two_word_label(self):
        line = f"{SUBJECT_FENS[0]} Checkmate Black\n"
        self.assertEqual(parse_chess_line(line, require_label=True), (SUBJECT_FENS[0], "Checkmate Black"))

    def test_one_word_label_and_extra_spaces(self):
        line = f"  {START_FEN}    Nothing  "
        self.assertEqual(parse_chess_line(line, require_label=True), (START_FEN, "Nothing"))

    def test_four_field_fen(self):
        fen = "8/8/8/8/8/8/8/k1K5 w - -"
        self.assertEqual(parse_chess_line(f"{fen} Nothing", require_label=True), (fen, "Nothing"))
        self.assertEqual(parse_chess_line(fen, require_label=False), (fen, None))

    def test_missing_label_in_train_mode_raises(self):
        with self.assertRaises(ValueError):
            parse_chess_line(START_FEN, require_label=True)

    def test_unknown_label_in_train_mode_raises(self):
        with self.assertRaises(ValueError):
            parse_chess_line(f"{START_FEN} Victory", require_label=True)

    def test_label_is_not_validated_in_predict_mode(self):
        self.assertEqual(parse_chess_line(f"{START_FEN} whatever", require_label=False), (START_FEN, "whatever"))

    def test_invalid_fens_raise(self):
        invalid = [
            "",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",          # 7 ranks
            "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # rank of 9 squares
            "rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   # rank of 7 squares
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1",  # unknown piece
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",  # bad turn
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkx - 0 1",  # bad castling
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w qkQK - 0 1",  # castling out of order
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e4 0 1", # bad en passant
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1", # negative clock
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0",    # 5 fields
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w",             # 2 fields
        ]
        for fen in invalid:
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                validate_fen(fen)

    def test_invalid_line_raises(self):
        with self.assertRaises(ValueError):
            parse_chess_line("not a fen at all", require_label=False)


class TestReadChessFile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name) / "boards.txt"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_reads_in_order_and_skips_blank_lines(self):
        lines = [f"{fen} {label}" for fen, label in zip(SUBJECT_FENS, SUBJECT_FULL_LABELS)]
        self.path.write_text("\n".join(lines[:2]) + "\n\n   \n" + "\n".join(lines[2:]) + "\n", encoding="utf-8")
        fens, labels = read_chess_file(str(self.path), require_label=True)
        self.assertEqual(fens, SUBJECT_FENS)
        self.assertEqual(labels, SUBJECT_FULL_LABELS)

    def test_error_mentions_line_number(self):
        self.path.write_text(f"{START_FEN}\n\nbroken/fen w - - 0 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r":3: "):
            read_chess_file(str(self.path), require_label=False)

    def test_file_without_boards_raises(self):
        self.path.write_text("\n  \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_chess_file(str(self.path), require_label=False)


class TestEncoding(unittest.TestCase):
    def test_shape_and_dtype(self):
        vector = encode_fen(START_FEN)
        self.assertEqual(vector.shape, (INPUT_SIZE,))
        self.assertEqual(INPUT_SIZE, 773)
        self.assertEqual(vector.dtype, np.float32)

    def test_start_position(self):
        vector = encode_fen(START_FEN)
        self.assertEqual(int(vector[:BOARD_FEATURES].sum()), 32)
        self.assertEqual(vector[feature_index("P", "e2")], 1.0)
        self.assertEqual(vector[feature_index("K", "e1")], 1.0)
        self.assertEqual(vector[feature_index("k", "e8")], 1.0)
        self.assertEqual(vector[feature_index("r", "a8")], 1.0)
        self.assertEqual(vector[feature_index("R", "h1")], 1.0)
        self.assertEqual(vector[feature_index("P", "e4")], 0.0)
        self.assertEqual(vector[BOARD_FEATURES], 1.0)
        np.testing.assert_array_equal(vector[BOARD_FEATURES + 1:], [1.0, 1.0, 1.0, 1.0])

    def test_subject_boards(self):
        matrix = encode_fens(SUBJECT_FENS)
        self.assertEqual(matrix.shape, (5, INPUT_SIZE))
        # Each square holds at most one piece.
        self.assertTrue(np.all(matrix[:, :BOARD_FEATURES].reshape(5, 12, 64).sum(axis=1) <= 1.0))
        np.testing.assert_array_equal(matrix[:, BOARD_FEATURES], [1.0, 1.0, 0.0, 0.0, 1.0])
        # Board 1: black queen on h4; board 3: white pawn moved d2 -> d4.
        self.assertEqual(matrix[0, feature_index("q", "h4")], 1.0)
        self.assertEqual(matrix[2, feature_index("P", "d4")], 1.0)
        self.assertEqual(matrix[2, feature_index("P", "d2")], 0.0)
        # Board 5: only two kings, no castling rights.
        self.assertEqual(int(matrix[4, :BOARD_FEATURES].sum()), 2)
        self.assertEqual(matrix[4, feature_index("k", "a1")], 1.0)
        self.assertEqual(matrix[4, feature_index("K", "c1")], 1.0)
        np.testing.assert_array_equal(matrix[4, BOARD_FEATURES + 1:], [0.0, 0.0, 0.0, 0.0])

    def test_partial_castling_rights(self):
        vector = encode_fen("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1")
        np.testing.assert_array_equal(vector[BOARD_FEATURES + 1:], [1.0, 0.0, 0.0, 1.0])


class TestLabels(unittest.TestCase):
    BASIC = ["Nothing", "Check"]
    MEDIUM = ["Nothing", "Check", "Checkmate"]

    def test_full_classes_keep_exact_labels(self):
        for label in FULL_CLASSES:
            self.assertEqual(adapt_label(label, FULL_CLASSES), label)

    def test_medium_drops_colors(self):
        self.assertEqual(adapt_label("Check White", self.MEDIUM), "Check")
        self.assertEqual(adapt_label("Checkmate Black", self.MEDIUM), "Checkmate")
        self.assertEqual(adapt_label("Nothing", self.MEDIUM), "Nothing")

    def test_basic_merges_checkmate_into_check(self):
        self.assertEqual(adapt_label("Checkmate Black", self.BASIC), "Check")
        self.assertEqual(adapt_label("Checkmate", self.BASIC), "Check")
        self.assertEqual(adapt_label("Check Black", self.BASIC), "Check")

    def test_stalemate(self):
        self.assertEqual(adapt_label("Stalemate", self.MEDIUM), "Nothing")
        self.assertEqual(adapt_label("Stalemate", FULL_CLASSES + ["Stalemate"]), "Stalemate")

    def test_coarse_label_cannot_fill_finer_classes(self):
        with self.assertRaises(ValueError):
            adapt_label("Check", FULL_CLASSES)

    def test_encode_labels_one_hot(self):
        targets = encode_labels(SUBJECT_FULL_LABELS, self.MEDIUM)
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_array_equal(
            targets,
            [[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
