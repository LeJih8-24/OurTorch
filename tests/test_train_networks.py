import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import unittest
import numpy as np

from analyzer.fen_encoder import INPUT_SIZE, encode_fens
from train_networks import encode_inputs, stratified_split


class TestTrainNetworks(unittest.TestCase):
    def test_stratified_split_is_disjoint_proportional_and_seeded(self):
        labels = ["Nothing"] * 500 + ["Check White"] * 300 + ["Checkmate Black"] * 200
        split = stratified_split(labels, seed=1)
        all_idx = np.concatenate(list(split.values()))
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(1000))
        self.assertEqual([len(split[p]) for p in ("train", "val", "test")], [800, 100, 100])
        test_labels = [labels[i] for i in split["test"]]
        self.assertEqual(test_labels.count("Nothing"), 50)
        self.assertEqual(test_labels.count("Checkmate Black"), 20)
        np.testing.assert_array_equal(split["test"], stratified_split(labels, seed=1)["test"])

    def test_uint8_encoding_matches_float_encoding(self):
        fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "8/8/8/8/8/8/8/k1K5 w - - 0 1"]
        X = encode_inputs(fens)
        self.assertEqual(X.dtype, np.uint8)
        self.assertEqual(X.shape, (2, INPUT_SIZE))
        np.testing.assert_array_equal(X, encode_fens(fens))

    def test_script_runs_on_a_small_slice(self):
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "train_networks.py"), "--levels", "basic",
                 "--per-file", "30", "--epochs", "1", "--no-save", "--out", out],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("basic: test accuracy", result.stdout)
            with open(os.path.join(out, "summary.json"), encoding="utf-8") as file:
                summary = json.load(file)
            self.assertEqual(summary["positions"], 270)
            self.assertEqual(summary["results"][0]["test_positions"], 27)
            self.assertTrue(os.path.exists(os.path.join(out, "basic_report.txt")))
            self.assertFalse(os.path.exists(os.path.join(out, "my_torch_network_basic.nn")))


if __name__ == "__main__":
    unittest.main()
