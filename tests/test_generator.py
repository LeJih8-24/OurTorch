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

from mytorch.serialization import load_network
from tests.test_config import VALID


class TestGenerator(unittest.TestCase):
    """Runs the generator as the real executable would: `make` copies scripts/generator.py
    to the repository root, next to the mytorch and analyzer packages."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_dir = tempfile.TemporaryDirectory()
        for package in ("mytorch", "analyzer"):
            shutil.copytree(os.path.join(ROOT, package), os.path.join(cls.bin_dir.name, package))
        cls.binary = os.path.join(cls.bin_dir.name, "my_torch_generator")
        shutil.copy(os.path.join(ROOT, "scripts", "generator.py"), cls.binary)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bin_dir.cleanup()

    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.conf = self.write_conf("basic_network.conf", VALID)

    def tearDown(self) -> None:
        self.work.cleanup()

    def write_conf(self, name: str, config) -> str:
        path = os.path.join(self.work.name, name)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file)
        return path

    def run_generator(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, self.binary, *args], cwd=self.work.name,
                              capture_output=True, text=True, timeout=60)

    def nn_files(self):
        return sorted(f for f in os.listdir(self.work.name) if f.endswith(".nn"))

    def test_generates_numbered_networks_in_cwd(self):
        result = self.run_generator("basic_network.conf", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.nn_files(), ["basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn"])
        self.assertEqual(result.stdout.split(), self.nn_files())

    def test_networks_differ_but_are_reproducible(self):
        self.run_generator("basic_network.conf", "2")
        first = [load_network(os.path.join(self.work.name, f)) for f in self.nn_files()]
        self.assertFalse(np.array_equal(first[0].layers[1].W, first[1].layers[1].W))
        self.assertEqual([m.seed for m in first], [42, 43])
        self.run_generator("basic_network.conf", "2")
        again = load_network(os.path.join(self.work.name, "basic_network_1.nn"))
        np.testing.assert_array_equal(again.layers[1].W, first[0].layers[1].W)

    def test_networks_differ_without_seed(self):
        config = dict(VALID, seed=None)
        self.write_conf("noseed.conf", config)
        self.run_generator("noseed.conf", "2")
        a, b = (load_network(os.path.join(self.work.name, f"noseed_{k}.nn")) for k in (1, 2))
        self.assertFalse(np.array_equal(a.layers[1].W, b.layers[1].W))

    def test_several_configs_and_repeated_name(self):
        self.write_conf("other.conf", VALID)
        result = self.run_generator("basic_network.conf", "1", "other.conf", "2", "basic_network.conf", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.nn_files(), ["basic_network_1.nn", "basic_network_2.nn", "other_1.nn", "other_2.nn"])

    def test_config_from_another_directory(self):
        result = self.run_generator(os.path.join(ROOT, "configs", "full_network.conf"), "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_network(os.path.join(self.work.name, "full_network_1.nn")).classes[-1], "Checkmate Black")

    def test_help(self):
        result = self.run_generator("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("USAGE", result.stdout)

    def test_errors_exit_84_and_write_nothing(self):
        self.write_conf("broken.conf", dict(VALID, classes=["only one"]))
        cases = {
            "no arguments": [],
            "odd arguments": ["basic_network.conf"],
            "zero": ["basic_network.conf", "0"],
            "negative": ["basic_network.conf", "-2"],
            "not a number": ["basic_network.conf", "two"],
            "missing config": ["missing.conf", "1"],
            "invalid config": ["broken.conf", "1"],
            "valid then invalid": ["basic_network.conf", "2", "broken.conf", "1"],
        }
        for name, args in cases.items():
            with self.subTest(name):
                result = self.run_generator(*args)
                self.assertEqual(result.returncode, 84)
                self.assertTrue(result.stderr.startswith("Error: "))
                self.assertEqual(result.stdout, "")
                self.assertEqual(self.nn_files(), [])


if __name__ == "__main__":
    unittest.main()
