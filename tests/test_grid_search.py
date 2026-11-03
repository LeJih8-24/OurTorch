import contextlib
import copy
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch import Sequential, build_from_config, grid_search, parameter_combinations
from mytorch.config import load_config, validate_config
from mytorch.model_selection import set_config_value
from tests.test_model_selection import SMALL_CONFIG, blobs

BASE = copy.deepcopy(SMALL_CONFIG)
BASE["training"] = {"epochs": 10, "batch_size": 32, "val_split": 0.2, "patience": 5}
BASE["layers"] = SMALL_CONFIG["layers"][:3] + [{"type": "dropout", "rate": 0.2}] + SMALL_CONFIG["layers"][3:]

# lr=1e-4 cannot learn in 10 epochs, lr=0.01 can: the search must find it
LR_GRID = {"optimizer.lr": [0.0001, 0.01]}


class TestParameterCombinations(unittest.TestCase):
    def test_cartesian_product_in_order(self):
        combos = parameter_combinations({"a": [1, 2], "b": ["x", "y", "z"]})
        self.assertEqual(len(combos), 6)
        self.assertEqual(combos[0], {"a": 1, "b": "x"})
        self.assertEqual(combos[-1], {"a": 2, "b": "z"})

    def test_invalid_grids(self):
        for grid in ({}, [], {"a": []}, {"a": 1}, {"a": "abc"}, {"": [1]}, {3: [1]}):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    parameter_combinations(grid)


class TestSetConfigValue(unittest.TestCase):
    def setUp(self) -> None:
        self.config = validate_config(BASE)

    def test_nested_paths(self):
        set_config_value(self.config, "optimizer.lr", 0.5)
        set_config_value(self.config, "training.batch_size", 8)
        set_config_value(self.config, "layers.1.out", 16)
        set_config_value(self.config, "seed", 3)
        self.assertEqual(self.config["optimizer"]["lr"], 0.5)
        self.assertEqual(self.config["training"]["batch_size"], 8)
        self.assertEqual(self.config["layers"][1]["out"], 16)
        self.assertEqual(self.config["seed"], 3)

    def test_wildcard_only_touches_layers_with_the_key(self):
        set_config_value(self.config, "layers.*.rate", 0.4)
        self.assertEqual([layer.get("rate") for layer in self.config["layers"]], [None, None, None, 0.4, None])

    def test_new_key_in_existing_dict(self):
        set_config_value(self.config, "optimizer.beta1", 0.8)
        self.assertEqual(self.config["optimizer"]["beta1"], 0.8)

    def test_whole_list_is_copied(self):
        layers = [{"type": "linear", "in": 2, "out": 2}]
        set_config_value(self.config, "layers", layers)
        layers[0]["out"] = 99
        self.assertEqual(self.config["layers"][0]["out"], 2)

    def test_invalid_paths(self):
        for path in ("nothing.lr", "layers.9.out", "layers.x.out", "layers.*", "layers.*.unknown",
                     "optimizer.lr.value"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    set_config_value(self.config, path, 1)


class TestGridSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = blobs()

    def test_finds_the_learning_rate_that_learns(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, seed=0)
        self.assertEqual(result["n_combinations"], 2)
        self.assertEqual(result["best_params"], {"optimizer.lr": 0.01})
        self.assertGreater(result["results"][0]["accuracy"], 0.9)
        self.assertLess(result["results"][0]["loss"], result["results"][1]["loss"])
        self.assertEqual(result["best_score"], result["results"][0]["loss"])
        self.assertEqual(result["best_config"]["optimizer"]["lr"], 0.01)
        self.assertIsNone(result["results"][0]["loss_std"])
        self.assertNotIn("best_model", result)

    def test_best_config_is_ready_to_build(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, seed=0)
        model = build_from_config(result["best_config"])
        self.assertEqual(model.optimizer.lr, 0.01)

    def test_accuracy_metric_ranks_highest_first(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, seed=0, metric="accuracy")
        scores = [r["score"] for r in result["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores, [r["accuracy"] for r in result["results"]])

    def test_cross_validation_mode(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, k=3, seed=0)
        best = result["results"][0]
        self.assertEqual(result["best_params"], {"optimizer.lr": 0.01})
        self.assertEqual(len(best["best_epochs"]), 3)
        self.assertIsNotNone(best["loss_std"])

    def test_explicit_validation_set(self):
        X_val, y_val = blobs(20, seed=1)
        result = grid_search(BASE, LR_GRID, self.X, self.y, validation_data=(X_val, y_val), seed=0, refit=True)
        self.assertEqual(result["best_params"], {"optimizer.lr": 0.01})
        self.assertIsInstance(result["best_model"], Sequential)
        self.assertGreater(result["best_model"].evaluate(X_val, y_val)[1], 0.9)

    def test_same_holdout_split_for_every_combination(self):
        # Two identical combinations must get exactly the same score.
        result = grid_search(BASE, {"optimizer.lr": [0.01, 0.01]}, self.X, self.y, seed=0)
        self.assertEqual(result["results"][0]["loss"], result["results"][1]["loss"])

    def test_reproducible(self):
        a = grid_search(BASE, LR_GRID, self.X, self.y, seed=4)
        b = grid_search(BASE, LR_GRID, self.X, self.y, seed=4)
        self.assertEqual([r["loss"] for r in a["results"]], [r["loss"] for r in b["results"]])

    def test_fit_overrides(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, seed=0, epochs=2, patience=None)
        self.assertTrue(all(r["best_epochs"][0] <= 2 for r in result["results"]))

    def test_refit_returns_trained_best_model(self):
        result = grid_search(BASE, LR_GRID, self.X, self.y, seed=0, refit=True)
        model = result["best_model"]
        self.assertEqual(model.optimizer.lr, 0.01)
        self.assertGreater(model.evaluate(self.X, self.y)[1], 0.9)

    def test_architecture_search_with_whole_layer_lists(self):
        small = [{"type": "normalization"}, {"type": "linear", "in": 2, "out": 2}]
        large = BASE["layers"]
        result = grid_search(BASE, {"layers": [small, large]}, self.X, self.y, seed=0)
        self.assertEqual(result["n_combinations"], 2)
        self.assertEqual({len(r["config"]["layers"]) for r in result["results"]}, {2, 5})

    def test_config_path(self):
        from analyzer.fen_encoder import INPUT_SIZE
        X = np.random.default_rng(0).random((60, INPUT_SIZE)).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.arange(60) % 2]
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "basic_network.conf")
        result = grid_search(path, {"layers.*.rate": [0.1, 0.3]}, X, y, seed=0, epochs=1)
        self.assertEqual(result["n_combinations"], 2)
        self.assertEqual(load_config(path)["layers"][3]["rate"], 0.2)  # file untouched

    def test_invalid_combination_fails_before_training(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaisesRegex(ValueError, r"Invalid combination layers\.1\.out=5"):
                grid_search(BASE, {"layers.1.out": [8, 5]}, self.X, self.y, seed=0, verbose=True)
        self.assertEqual(err.getvalue(), "")  # nothing was trained

    def test_verbose_logs_on_stderr_only(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            grid_search(BASE, LR_GRID, self.X, self.y, seed=0, epochs=1, verbose=True)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("[2/2]", err.getvalue())
        self.assertIn("best", err.getvalue())

    def test_errors(self):
        cases = {
            "unknown metric": dict(metric="f1"),
            "k and validation_data": dict(k=3, validation_data=blobs(10)),
            "bad val_split": dict(val_split=1.0),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    grid_search(BASE, LR_GRID, self.X, self.y, **kwargs)
        with self.assertRaises(ValueError):
            grid_search(BASE, LR_GRID, self.X, self.y[:5])
        with self.assertRaises(ValueError):
            grid_search(dict(BASE, classes=["a"]), LR_GRID, self.X, self.y)


if __name__ == "__main__":
    unittest.main()
