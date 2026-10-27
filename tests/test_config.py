import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.config import build_from_config, load_config, validate_config
from mytorch.layers import Dropout, Linear, Normalization, ReLU
from mytorch.losses import SoftmaxCrossEntropyLoss
from mytorch.optimizers import Adam

VALID = {
    "classes": ["Nothing", "Check", "Checkmate"],
    "loss": "cross_entropy",
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 1e-4},
    "training": {"epochs": 10, "batch_size": 64},
    "seed": 42,
    "layers": [
        {"type": "normalization"},
        {"type": "linear", "in": 20, "out": 16, "init": "he"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "in": 16, "out": 3, "init": "xavier"},
    ],
}


def with_changes(path, value):
    """Copy of VALID with config[path...] = value (value=KeyError deletes the key)."""
    config = copy.deepcopy(VALID)
    target = config
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return config


class TestValidateConfig(unittest.TestCase):
    def test_defaults_filled(self):
        config = validate_config({"classes": ["a", "b"], "layers": [{"type": "linear", "in": 4, "out": 2}]})
        self.assertEqual(config["loss"], "cross_entropy")
        self.assertEqual(config["optimizer"], {"type": "adam"})
        self.assertEqual(config["training"], {"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5})
        self.assertIsNone(config["seed"])
        self.assertEqual(config["layers"][0]["init"], "he")

    def test_normalization_size_is_inferred(self):
        config = validate_config(VALID)
        self.assertEqual(config["layers"][0], {"type": "normalization", "features": 20})
        self.assertEqual(config["layers"][3], {"type": "dropout", "rate": 0.3, "mc_dropout": False})

    def test_does_not_modify_input(self):
        original = copy.deepcopy(VALID)
        validate_config(VALID)
        self.assertEqual(VALID, original)

    def test_invalid_configs(self):
        cases = {
            "not an object": [1, 2],
            "missing classes": with_changes(["classes"], KeyError),
            "missing layers": with_changes(["layers"], KeyError),
            "unknown top-level key": with_changes(["learning_rate"], 0.1),
            "one class": with_changes(["classes"], ["Nothing"]),
            "duplicate classes": with_changes(["classes"], ["a", "a", "b"]),
            "unknown loss": with_changes(["loss"], "hinge"),
            "unknown optimizer": with_changes(["optimizer"], {"type": "rmsprop"}),
            "bad optimizer param": with_changes(["optimizer", "lr"], -1),
            "empty layers": with_changes(["layers"], []),
            "unknown layer": with_changes(["layers", 2], {"type": "gelu"}),
            "layer without type": with_changes(["layers", 2], {"alpha": 1}),
            "unknown layer key": with_changes(["layers", 2], {"type": "relu", "inplace": True}),
            "linear missing out": with_changes(["layers", 1], {"type": "linear", "in": 20}),
            "linear negative size": with_changes(["layers", 1, "out"], -16),
            "linear bool size": with_changes(["layers", 1, "in"], True),
            "unknown init": with_changes(["layers", 1, "init"], "zeros"),
            "dimension mismatch": with_changes(["layers", 4, "in"], 32),
            "output != classes": with_changes(["layers", 4, "out"], 5),
            "normalization size mismatch": with_changes(["layers", 0, "features"], 19),
            "normalization not first": with_changes(["layers", 2], {"type": "normalization"}),
            "dropout rate 1": with_changes(["layers", 3, "rate"], 1.0),
            "no linear layer": with_changes(["layers"], [{"type": "relu"}]),
            "softmax + cross entropy": with_changes(["layers"], VALID["layers"] + [{"type": "softmax"}]),
            "zero epochs": with_changes(["training", "epochs"], 0),
            "bad val_split": with_changes(["training", "val_split"], 1.0),
            "unknown training key": with_changes(["training", "lr"], 0.1),
            "negative seed": with_changes(["seed"], -1),
            "float seed": with_changes(["seed"], 1.5),
        }
        for name, config in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_error_message_points_to_field(self):
        with self.assertRaisesRegex(ValueError, r"layers\[4\]\.in is 32"):
            validate_config(with_changes(["layers", 4, "in"], 32))


class TestBuildFromConfig(unittest.TestCase):
    def test_builds_expected_model(self):
        model = build_from_config(VALID)
        types = [type(layer) for layer in model.layers]
        self.assertEqual(types, [Normalization, Linear, ReLU, Dropout, Linear])
        self.assertIsInstance(model.loss, SoftmaxCrossEntropyLoss)
        self.assertIsInstance(model.optimizer, Adam)
        self.assertEqual(model.optimizer.weight_decay, 1e-4)
        self.assertEqual(model.classes, VALID["classes"])
        self.assertEqual(model.training_config["batch_size"], 64)
        self.assertEqual(model.seed, 42)

    def test_seed_makes_weights_reproducible(self):
        w1 = build_from_config(VALID).layers[1].W
        w2 = build_from_config(VALID).layers[1].W
        np.testing.assert_array_equal(w1, w2)
        w3 = build_from_config(with_changes(["seed"], 43)).layers[1].W
        self.assertFalse(np.array_equal(w1, w3))

    def test_get_config_round_trip(self):
        model = build_from_config(VALID)
        rebuilt = build_from_config(model.get_config())
        self.assertEqual(rebuilt.get_config(), model.get_config())
        np.testing.assert_array_equal(rebuilt.layers[1].W, model.layers[1].W)

    def test_trains_with_its_training_config(self):
        model = build_from_config(VALID)
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 100)]
        history = model.fit(X, y, **model.training_config)
        self.assertTrue(model.layers[0].adapted)
        self.assertGreater(len(history["train_loss"]), 0)


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, content: str) -> str:
        path = os.path.join(self.tmp.name, "net.conf")
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return path

    def test_load_and_build_from_path(self):
        path = self.write(json.dumps(VALID))
        self.assertEqual(load_config(path)["layers"][0]["features"], 20)
        self.assertEqual(len(build_from_config(path).layers), 5)

    def test_file_errors(self):
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            load_config(os.path.join(self.tmp.name, "missing.conf"))
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            load_config(self.write("{ not json"))
        with self.assertRaisesRegex(ValueError, "Invalid configuration file"):
            load_config(self.write(json.dumps({"classes": ["a"]})))


if __name__ == "__main__":
    unittest.main()
