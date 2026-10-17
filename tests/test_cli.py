import contextlib
import io
from pathlib import Path
import tempfile
import unittest

try:
    import pytest
except ImportError:
    pytest = None

from analyzer.cli import AnalyzerArgs, HELP_TEXT, parse_args


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.loadfile = self.tmp_path / "model.nn"
        self.chessfile = self.tmp_path / "chess.txt"
        self.loadfile.write_text("dummy model content\n", encoding="utf-8")
        self.chessfile.write_text("dummy fen content\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _assert_exit(self, argv, expected_code: int):
        out = io.StringIO()
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                parse_args(argv)
        self.assertEqual(ctx.exception.code, expected_code)
        return out.getvalue(), err.getvalue()

    def test_help_short(self):
        out, err = self._assert_exit(["-h"], 0)
        self.assertEqual(out.strip(), HELP_TEXT.strip())
        self.assertEqual(err, "")

    def test_help_long(self):
        out, err = self._assert_exit(["--help"], 0)
        self.assertEqual(out.strip(), HELP_TEXT.strip())
        self.assertEqual(err, "")

    def test_no_args_exits_84(self):
        _, err = self._assert_exit([], 84)
        self.assertIn("Error:", err)

    def test_valid_predict(self):
        args = parse_args(["--predict", str(self.loadfile), str(self.chessfile)])
        self.assertEqual(
            args,
            AnalyzerArgs(
                mode="predict",
                loadfile=str(self.loadfile),
                chessfile=str(self.chessfile),
                savefile=None,
            ),
        )

    def test_valid_train_without_save(self):
        args = parse_args(["--train", str(self.loadfile), str(self.chessfile)])
        self.assertEqual(
            args,
            AnalyzerArgs(
                mode="train",
                loadfile=str(self.loadfile),
                chessfile=str(self.chessfile),
                savefile=None,
            ),
        )

    def test_valid_train_with_save(self):
        args = parse_args(
            ["--train", "--save", "out.nn", str(self.loadfile), str(self.chessfile)]
        )
        self.assertEqual(
            args,
            AnalyzerArgs(
                mode="train",
                loadfile=str(self.loadfile),
                chessfile=str(self.chessfile),
                savefile="out.nn",
            ),
        )

    def test_predict_with_save_exits_84(self):
        _, err = self._assert_exit(
            ["--predict", "--save", "out.nn", str(self.loadfile), str(self.chessfile)],
            84,
        )
        self.assertIn("Error:", err)

    def test_both_modes_exits_84(self):
        _, err = self._assert_exit(
            ["--train", "--predict", str(self.loadfile), str(self.chessfile)],
            84,
        )
        self.assertIn("Error:", err)

    def test_missing_mode_exits_84(self):
        _, err = self._assert_exit([str(self.loadfile), str(self.chessfile)], 84)
        self.assertIn("Error:", err)

    def test_save_missing_argument_exits_84(self):
        _, err = self._assert_exit(
            ["--train", str(self.loadfile), str(self.chessfile), "--save"],
            84,
        )
        self.assertIn("Error:", err)

    def test_missing_positional_files_exits_84(self):
        _, err = self._assert_exit(["--predict", str(self.loadfile)], 84)
        self.assertIn("Error:", err)

    def test_extra_positional_files_exits_84(self):
        _, err = self._assert_exit(
            ["--predict", str(self.loadfile), str(self.chessfile), "extra.txt"],
            84,
        )
        self.assertIn("Error:", err)

    def test_unknown_flag_exits_84(self):
        _, err = self._assert_exit(
            ["--predict", "--unknown", str(self.loadfile), str(self.chessfile)],
            84,
        )
        self.assertIn("Error:", err)

    def test_nonexistent_file_exits_84(self):
        _, err = self._assert_exit(
            ["--predict", "nonexistent.nn", str(self.chessfile)],
            84,
        )
        self.assertIn("Error:", err)

    def test_empty_file_exits_84(self):
        empty_load = self.tmp_path / "empty.nn"
        empty_load.write_text("", encoding="utf-8")
        _, err = self._assert_exit(
            ["--predict", str(empty_load), str(self.chessfile)],
            84,
        )
        self.assertIn("Error:", err)


if __name__ == "__main__":
    unittest.main()
