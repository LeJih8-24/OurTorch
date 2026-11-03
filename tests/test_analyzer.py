import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import unittest
import numpy as np

from analyzer.fen_encoder import INPUT_SIZE, adapt_label, encode_fens
from mytorch.config import build_from_config
from mytorch.serialization import load_network, read_metadata, save_network

# Small but real positions from the dataset (10 of each label family)
DATASET_FILES = ["check/20_pieces.txt", "nothing/20_pieces.txt", "checkmate/20_pieces.txt"]

TINY_CONFIG = {
    "classes": ["Nothing", "Check", "Checkmate"],
    "optimizer": {"type": "adam", "lr": 0.01},
    "training": {"epochs": 60, "batch_size": 8, "val_split": 0.0, "patience": None},
    "seed": 0,
    "layers": [
        {"type": "normalization"},
        {"type": "linear", "in": INPUT_SIZE, "out": 16},
        {"type": "relu"},
        {"type": "linear", "in": 16, "out": 3, "init": "xavier"},
    ],
}


def dataset_lines(per_file: int = 10):
    lines = []
    for name in DATASET_FILES:
        with open(os.path.join(ROOT, "dataset", name), encoding="utf-8") as file:
            lines += [next(file).strip() for _ in range(per_file)]
    return lines


class TestAnalyzer(unittest.TestCase):
    """Runs my_torch_analyzer as the real executable: `make` copies analyzer/main.py
    to the repository root, next to the mytorch and analyzer packages."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_dir = tempfile.TemporaryDirectory()
        for package in ("mytorch", "analyzer"):
            shutil.copytree(os.path.join(ROOT, package), os.path.join(cls.bin_dir.name, package))
        cls.analyzer = os.path.join(cls.bin_dir.name, "my_torch_analyzer")
        cls.generator = os.path.join(cls.bin_dir.name, "my_torch_generator")
        shutil.copy(os.path.join(ROOT, "analyzer", "main.py"), cls.analyzer)
        shutil.copy(os.path.join(ROOT, "scripts", "generator.py"), cls.generator)
        cls.lines = dataset_lines()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bin_dir.cleanup()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.network = self.path("net.nn")
        save_network(build_from_config(TINY_CONFIG), self.network)
        self.train_file = self.write("train.txt", self.lines)
        self.fens = [" ".join(line.split()[:6]) for line in self.lines]
        self.predict_file = self.write("predict.txt", self.fens)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def write(self, name: str, lines) -> str:
        with open(self.path(name), "w", encoding="utf-8") as file:
            file.write("".join(f"{line}\n" for line in lines))
        return self.path(name)

    def run_analyzer(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, self.analyzer, *args], cwd=self.tmp.name,
                              capture_output=True, text=True, timeout=120)

    def assert_error(self, result: subprocess.CompletedProcess, fragment: str = "") -> None:
        self.assertEqual(result.returncode, 84, result.stderr)
        self.assertTrue(result.stderr.startswith("Error: "), result.stderr)
        self.assertIn(fragment, result.stderr)
        self.assertEqual(result.stdout, "")

    def expected(self, classes):
        return [adapt_label(" ".join(line.split()[6:]), classes) for line in self.lines]

    # ----- predict ---------------------------------------------------------------

    def test_predict_one_line_per_position_in_order(self):
        result = self.run_analyzer("--predict", self.network, self.predict_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        predictions = result.stdout.splitlines()
        self.assertEqual(len(predictions), len(self.fens))
        self.assertTrue(set(predictions) <= set(TINY_CONFIG["classes"]))
        # Same order as the input: predicting each position alone gives the same lines
        model = load_network(self.network)
        self.assertEqual(predictions, model.predict(encode_fens(self.fens)))

    def test_predict_ignores_expected_outputs_and_blank_lines(self):
        with_labels = self.run_analyzer("--predict", self.network, self.train_file)
        spaced = self.write("spaced.txt", [self.fens[0], "", self.fens[1], "   "])
        result = self.run_analyzer("--predict", self.network, spaced)
        self.assertEqual(with_labels.returncode, 0, with_labels.stderr)
        plain = self.run_analyzer("--predict", self.network, self.predict_file)
        self.assertEqual(with_labels.stdout, plain.stdout)
        self.assertEqual(len(result.stdout.splitlines()), 2)

    def test_predict_more_positions_than_one_chunk(self):
        many = self.write("many.txt", self.fens * 40)  # 1200 positions > 1024
        result = self.run_analyzer("--predict", self.network, many)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), self.run_analyzer(
            "--predict", self.network, self.predict_file).stdout.splitlines() * 40)

    # ----- train -----------------------------------------------------------------

    def test_train_overwrites_loadfile_without_save(self):
        result = self.run_analyzer("--train", self.network, self.train_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Network saved to", result.stdout)
        self.assertIn("epoch 60/60", result.stderr)  # training logs on stderr
        model = load_network(self.network)
        self.assertTrue(model.layers[0].adapted)
        self.assertEqual(read_metadata(self.network)["trained_on"], len(self.lines))

    def test_train_with_save_keeps_loadfile(self):
        with open(self.network, "rb") as file:
            before = file.read()
        result = self.run_analyzer("--train", "--save", "out.nn", self.network, self.train_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.network, "rb") as file:
            self.assertEqual(file.read(), before)
        self.assertTrue(load_network(self.path("out.nn")).layers[0].adapted)

    def test_train_then_predict_learns_the_training_positions(self):
        self.run_analyzer("--train", "--save", "out.nn", self.network, self.train_file)
        result = self.run_analyzer("--predict", self.path("out.nn"), self.predict_file)
        predictions = result.stdout.splitlines()
        expected = self.expected(TINY_CONFIG["classes"])
        accuracy = np.mean([p == e for p, e in zip(predictions, expected)])
        self.assertGreaterEqual(accuracy, 0.9)

    def test_train_is_reproducible(self):
        self.run_analyzer("--train", "--save", "a.nn", self.network, self.train_file)
        self.run_analyzer("--train", "--save", "b.nn", self.network, self.train_file)
        a, b = load_network(self.path("a.nn")), load_network(self.path("b.nn"))
        np.testing.assert_array_equal(a.layers[1].W, b.layers[1].W)

    def test_full_chain_with_generator(self):
        """Acceptance test of issue #1: generate, train, predict."""
        conf = self.path("tiny.conf")
        with open(conf, "w", encoding="utf-8") as file:
            json.dump(TINY_CONFIG, file)
        generated = subprocess.run([sys.executable, self.generator, conf, "1"], cwd=self.tmp.name,
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        trained = self.run_analyzer("--train", "--save", "trained.nn", "tiny_1.nn", self.train_file)
        self.assertEqual(trained.returncode, 0, trained.stderr)
        predicted = self.run_analyzer("--predict", "trained.nn", self.predict_file)
        self.assertEqual(predicted.returncode, 0, predicted.stderr)
        self.assertEqual(len(predicted.stdout.splitlines()), len(self.fens))

    def test_basic_network_outputs_only_its_classes(self):
        config = dict(TINY_CONFIG, classes=["Nothing", "Check"])
        config["layers"] = TINY_CONFIG["layers"][:3] + [{"type": "linear", "in": 16, "out": 2}]
        save_network(build_from_config(config), self.network)
        self.assertEqual(self.run_analyzer("--train", self.network, self.train_file).returncode, 0)
        result = self.run_analyzer("--predict", self.network, self.predict_file)
        self.assertTrue(set(result.stdout.splitlines()) <= {"Nothing", "Check"})

    # ----- errors ----------------------------------------------------------------

    def test_corrupted_network(self):
        corrupted = self.write("corrupted.nn", ["not a network"])
        self.assert_error(self.run_analyzer("--predict", corrupted, self.predict_file), "not a valid network file")

    def test_invalid_fen_reports_the_line(self):
        bad = self.write("bad.txt", [self.fens[0], "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"])
        self.assert_error(self.run_analyzer("--predict", self.network, bad), "bad.txt:2")

    def test_train_requires_labels(self):
        self.assert_error(self.run_analyzer("--train", self.network, self.predict_file), "Missing expected output")

    def test_train_rejects_unknown_labels(self):
        unknown = self.write("unknown.txt", [f"{self.fens[0]} Draw"])
        self.assert_error(self.run_analyzer("--train", self.network, unknown), "Unknown label")

    def test_network_with_wrong_input_size(self):
        config = json.loads(json.dumps(TINY_CONFIG))
        config["layers"][1]["in"] = 789
        save_network(build_from_config(config), self.network)
        self.assert_error(self.run_analyzer("--predict", self.network, self.predict_file), "expects 789 input features")

    def test_unwritable_savefile_keeps_loadfile(self):
        with open(self.network, "rb") as file:
            before = file.read()
        result = self.run_analyzer("--train", "--save", "missing_dir/out.nn", self.network, self.train_file)
        self.assertEqual(result.returncode, 84)
        self.assertIn("Cannot write", result.stderr)
        with open(self.network, "rb") as file:
            self.assertEqual(file.read(), before)

    def test_cli_errors_still_exit_84(self):
        self.assert_error(self.run_analyzer("--predict", "--save", "x.nn", self.network, self.predict_file))
        self.assert_error(self.run_analyzer(self.network, self.predict_file))

    def test_help(self):
        result = self.run_analyzer("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("USAGE", result.stdout)


if __name__ == "__main__":
    unittest.main()
