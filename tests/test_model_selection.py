import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch import Adam, Sequential, SoftmaxCrossEntropyLoss
from mytorch.layers import Linear, Normalization, ReLU
from mytorch.model_selection import cross_validate, kfold_indices

SMALL_CONFIG = {
    "classes": ["a", "b"],
    "optimizer": {"type": "adam", "lr": 0.01},
    "training": {"epochs": 15, "batch_size": 32, "val_split": 0.15, "patience": 5},
    "seed": 0,
    "layers": [
        {"type": "normalization"},
        {"type": "linear", "in": 2, "out": 8},
        {"type": "relu"},
        {"type": "linear", "in": 8, "out": 2, "init": "xavier"},
    ],
}


def blobs(n_per_class: int = 100, seed: int = 0):
    """Two well separated Gaussian blobs: an easy 2-class problem."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2, 1, (n_per_class, 2)), rng.normal(2, 1, (n_per_class, 2))]).astype(np.float32)
    y = np.repeat(np.eye(2, dtype=np.float32), n_per_class, axis=0)
    return X, y


class TestKFoldIndices(unittest.TestCase):
    def test_every_sample_validated_exactly_once(self):
        folds = kfold_indices(103, k=5, seed=0)
        self.assertEqual(len(folds), 5)
        all_val = np.concatenate([val for _, val in folds])
        np.testing.assert_array_equal(np.sort(all_val), np.arange(103))
        for train, val in folds:
            self.assertEqual(len(np.intersect1d(train, val)), 0)
            self.assertEqual(len(train) + len(val), 103)
        sizes = [len(val) for _, val in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_stratified_keeps_class_proportions(self):
        labels = np.array([0] * 90 + [1] * 7 + [2] * 3)
        one_hot = np.eye(3)[labels]
        for stratify in (labels, one_hot):
            with self.subTest(kind="indices" if stratify.ndim == 1 else "one-hot"):
                folds = kfold_indices(100, k=5, stratify=stratify, seed=1)
                for _, val in folds:
                    counts = np.bincount(labels[val], minlength=3)
                    self.assertEqual(counts[0], 18)
                    self.assertIn(counts[1], (1, 2))
                    self.assertIn(counts[2], (0, 1))
                sizes = [len(val) for _, val in folds]
                self.assertEqual(sizes, [20] * 5)

    def test_rare_class_spread_over_folds(self):
        labels = np.array([0] * 45 + [1] * 5)
        folds = kfold_indices(50, k=5, stratify=labels, seed=0)
        self.assertEqual([int(np.sum(labels[val] == 1)) for _, val in folds], [1] * 5)

    def test_reproducible(self):
        a = kfold_indices(50, 5, seed=3)
        b = kfold_indices(50, 5, seed=3)
        for (ta, va), (tb, vb) in zip(a, b):
            np.testing.assert_array_equal(va, vb)
        np.random.seed(7)
        c = kfold_indices(50, 5)
        np.random.seed(7)
        d = kfold_indices(50, 5)
        np.testing.assert_array_equal(c[0][1], d[0][1])
        self.assertFalse(np.array_equal(a[0][1], kfold_indices(50, 5, seed=4)[0][1]))

    def test_no_shuffle_gives_contiguous_folds(self):
        folds = kfold_indices(10, k=3, shuffle=False)
        self.assertEqual([val.tolist() for _, val in folds], [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_invalid_arguments(self):
        for kwargs in ({"n_samples": 10, "k": 1}, {"n_samples": 10, "k": 2.5}, {"n_samples": 10, "k": True},
                       {"n_samples": 3, "k": 5}, {"n_samples": 10, "k": 2, "stratify": np.zeros(9)},
                       {"n_samples": 4, "k": 2, "stratify": np.zeros((2, 2, 1))}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    kfold_indices(**kwargs)


class TestCrossValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = blobs()

    def test_scores_on_easy_problem(self):
        result = cross_validate(SMALL_CONFIG, self.X, self.y, k=4, seed=0)
        self.assertEqual(len(result["fold_acc"]), 4)
        self.assertEqual(len(result["histories"]), 4)
        self.assertGreater(result["mean_acc"], 0.9)
        self.assertAlmostEqual(result["mean_acc"], float(np.mean(result["fold_acc"])))
        self.assertAlmostEqual(result["std_loss"], float(np.std(result["fold_loss"])))
        self.assertTrue(all(epoch is not None for epoch in result["best_epochs"]))

    def test_fit_overrides_replace_config_settings(self):
        result = cross_validate(SMALL_CONFIG, self.X, self.y, k=3, seed=0, epochs=2, patience=None)
        self.assertEqual([len(h["train_loss"]) for h in result["histories"]], [2, 2, 2])

    def test_reproducible(self):
        a = cross_validate(SMALL_CONFIG, self.X, self.y, k=3, seed=5, epochs=3)
        b = cross_validate(SMALL_CONFIG, self.X, self.y, k=3, seed=5, epochs=3)
        self.assertEqual(a["fold_loss"], b["fold_loss"])

    def test_fresh_network_and_statistics_per_fold(self):
        # Shift one half of the data: each fold must compute its normalization
        # statistics on its own training part, never reuse those of another fold.
        built = []

        def factory():
            model = Sequential([Normalization(2), Linear(2, 4), ReLU(), Linear(4, 2)],
                               SoftmaxCrossEntropyLoss(), Adam(lr=0.01), ["a", "b"])
            built.append(model)
            return model

        X = self.X.copy()
        X[:100] += 50.0
        result = cross_validate(factory, X, self.y, k=4, seed=0, stratified=False,
                                epochs=1, batch_size=32, val_split=0.0)
        self.assertEqual(len(built), 4)
        self.assertEqual(len({id(m) for m in built}), 4)
        for model, (train_idx, _) in zip(built, result["folds"]):
            np.testing.assert_allclose(model.layers[0].mean, X[train_idx].mean(axis=0, keepdims=True), rtol=1e-4)

    def test_config_path(self):
        from analyzer.fen_encoder import INPUT_SIZE
        X = np.random.default_rng(0).random((40, INPUT_SIZE)).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.arange(40) % 2]
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "basic_network.conf")
        result = cross_validate(path, X, y, k=2, seed=0, epochs=1)
        self.assertEqual(len(result["fold_loss"]), 2)

    def test_verbose_logs_on_stderr_only(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cross_validate(SMALL_CONFIG, self.X, self.y, k=2, seed=0, epochs=1, verbose=True)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("fold 2/2", err.getvalue())
        self.assertIn("cross-validation", err.getvalue())

    def test_errors(self):
        with self.assertRaises(ValueError):
            cross_validate(SMALL_CONFIG, self.X, self.y[:10], k=2)
        with self.assertRaises(ValueError):
            cross_validate(SMALL_CONFIG, self.X[0], self.y, k=2)
        with self.assertRaises(ValueError):
            cross_validate(lambda: "not a model", self.X, self.y, k=2)
        bare = lambda: Sequential([Linear(2, 2)], SoftmaxCrossEntropyLoss(), Adam(), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "epochs"):
            cross_validate(bare, self.X, self.y, k=2)
        with self.assertRaises(ValueError):
            cross_validate(dict(SMALL_CONFIG, classes=["a"]), self.X, self.y, k=2)


if __name__ == "__main__":
    unittest.main()
