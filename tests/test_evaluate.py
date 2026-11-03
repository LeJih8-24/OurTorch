import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import unittest

from mytorch.config import build_from_config
from mytorch.serialization import save_network
from tests.test_analyzer import TINY_CONFIG, dataset_lines

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import evaluate  # noqa: E402


class TestEvaluate(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self):
        report = evaluate(["Nothing", "Check"], ["Nothing", "Check", "Check", "Nothing"],
                          ["Nothing", "Check", "Nothing", "Nothing"])
        self.assertIn("accuracy 0.7500 (3/4)", report)
        lines = report.splitlines()
        self.assertTrue(any(line.startswith("Check") and "0.500   1.000" in line for line in lines))
        self.assertEqual(lines[-2].split()[-2:], ["2", "1"])   # Nothing: 2 right, 1 predicted Check
        self.assertEqual(lines[-1].split()[-2:], ["0", "1"])

    def test_unknown_prediction(self):
        with self.assertRaises(ValueError):
            evaluate(["Nothing", "Check"], ["Checkmate"], ["Nothing"])

    def test_script_end_to_end_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            network = os.path.join(tmp, "net.nn")
            save_network(build_from_config(TINY_CONFIG), network)
            chessfile = os.path.join(tmp, "test.txt")
            lines = dataset_lines(2)
            with open(chessfile, "w", encoding="utf-8") as file:
                file.write("".join(f"{line}\n" for line in lines))
            predictions = os.path.join(tmp, "pred.txt")
            with open(predictions, "w", encoding="utf-8") as file:
                file.write("Nothing\n" * len(lines))
            script = os.path.join(ROOT, "scripts", "evaluate.py")
            ok = subprocess.run([sys.executable, script, network, predictions, chessfile],
                                capture_output=True, text=True, timeout=60)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("accuracy 0.3333 (2/6)", ok.stdout)
            with open(predictions, "w", encoding="utf-8") as file:
                file.write("Nothing\n")
            bad = subprocess.run([sys.executable, script, network, predictions, chessfile],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(bad.returncode, 84)
            self.assertIn("1 predictions for 6 positions", bad.stderr)
            usage = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
            self.assertEqual(usage.returncode, 84)


if __name__ == "__main__":
    unittest.main()
