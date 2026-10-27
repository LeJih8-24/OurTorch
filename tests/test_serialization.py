import copy
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.config import build_from_config
from mytorch.serialization import load_network, read_metadata, save_network
from tests.test_config import VALID


def trained_model():
    np.random.seed(0)
    model = build_from_config(VALID)
    X = np.random.randn(120, 20).astype(np.float32) * 5 + 3
    y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 120)]
    model.fit(X, y, epochs=3, batch_size=32, val_split=0.0)
    return model, X


class TestSaveLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "net.nn")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def rewrite(self, **changes) -> None:
        """Rewrite the saved archive with some entries replaced (None removes an entry)."""
        with np.load(self.path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        for name, value in changes.items():
            if value is None:
                arrays.pop(name)
            else:
                arrays[name] = value
        with open(self.path, "wb") as file:
            np.savez(file, **arrays)

    def test_round_trip_gives_identical_predictions(self):
        model, X = trained_model()
        save_network(model, self.path)
        loaded = load_network(self.path)
        self.assertEqual(loaded.get_config(), model.get_config())
        np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
        self.assertEqual(loaded.predict(X), model.predict(X))

    def test_normalization_statistics_are_saved(self):
        model, _ = trained_model()
        save_network(model, self.path)
        norm = load_network(self.path).layers[0]
        self.assertTrue(norm.adapted)
        np.testing.assert_array_equal(norm.mean, model.layers[0].mean)

    def test_blank_network_stays_blank(self):
        save_network(build_from_config(VALID), self.path)
        self.assertFalse(load_network(self.path).layers[0].adapted)

    def test_exact_file_name_and_overwrite(self):
        model, _ = trained_model()
        save_network(model, self.path)
        save_network(build_from_config(VALID), self.path)
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["net.nn"])  # no .npz suffix, no temp file left
        self.assertFalse(load_network(self.path).layers[0].adapted)

    def test_loaded_network_can_be_retrained(self):
        model, X = trained_model()
        save_network(model, self.path)
        loaded = load_network(self.path)
        y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, len(X))]
        loaded.fit(X, y, epochs=2, batch_size=32, val_split=0.0)
        self.assertFalse(np.array_equal(loaded.layers[1].W, model.layers[1].W))

    def test_file_permissions_follow_umask(self):
        save_network(build_from_config(VALID), self.path)
        umask = os.umask(0)
        os.umask(umask)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o666 & ~umask)

    def test_metadata(self):
        save_network(build_from_config(VALID), self.path, metadata={"encoder": "fen-v1", "input_size": 20})
        self.assertEqual(read_metadata(self.path), {"encoder": "fen-v1", "input_size": 20})

    def test_file_errors(self):
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            load_network(os.path.join(self.tmp.name, "missing.nn"))
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("dummy model content\n")
        with self.assertRaisesRegex(ValueError, "not a valid network file"):
            load_network(self.path)
        with self.assertRaisesRegex(ValueError, "Cannot write"):
            save_network(build_from_config(VALID), os.path.join(self.tmp.name, "no_dir", "net.nn"))

    def test_truncated_file(self):
        save_network(build_from_config(VALID), self.path)
        with open(self.path, "rb") as file:
            content = file.read()
        with open(self.path, "wb") as file:
            file.write(content[: len(content) // 2])
        with self.assertRaises(ValueError):
            load_network(self.path)

    def test_plain_npy_file_is_rejected(self):
        np.save(self.path + ".npy", np.zeros(3))
        os.replace(self.path + ".npy", self.path)
        with self.assertRaisesRegex(ValueError, "not a valid network file"):
            load_network(self.path)

    def test_corrupted_contents(self):
        bad_config = copy.deepcopy(VALID)
        bad_config["layers"][4]["out"] = 4
        cases = {
            "missing header": {"header": None},
            "wrong format": {"header": np.array(json.dumps({"format": "other", "version": 1}))},
            "future version": {"header": np.array(json.dumps({"format": "mytorch-nn", "version": 2}))},
            "missing config": {"config": None},
            "config not JSON": {"config": np.array("{oops")},
            "invalid config": {"config": np.array(json.dumps(bad_config))},
            "missing weights": {"layer1.W": None},
            "wrong shape": {"layer1.W": np.zeros((3, 3), dtype=np.float32)},
            "non numeric": {"layer1.b": np.array(["a"] * 16).reshape(1, 16)},
            "unexpected array": {"layer9.W": np.zeros(2)},
        }
        for name, changes in cases.items():
            with self.subTest(name):
                save_network(build_from_config(VALID), self.path)
                self.rewrite(**changes)
                with self.assertRaises(ValueError):
                    load_network(self.path)

    def test_pickled_object_is_never_loaded(self):
        save_network(build_from_config(VALID), self.path)
        self.rewrite(**{"layer1.W": np.array([{"evil": 1}], dtype=object)})
        with self.assertRaises(ValueError):
            load_network(self.path)
        self.assertTrue(zipfile.is_zipfile(self.path))


if __name__ == "__main__":
    unittest.main()
