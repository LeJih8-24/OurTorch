import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.layers import Dropout, Linear, ReLU
from mytorch.losses import MSELoss, SoftmaxCrossEntropyLoss
from mytorch.model import Sequential
from mytorch.optimizers import SGD, Adam

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
XOR_CLASSES = ["0", "1"]


def xor_model(optimizer) -> Sequential:
    return Sequential([Linear(2, 8), ReLU(), Linear(8, 2)], SoftmaxCrossEntropyLoss(), optimizer, XOR_CLASSES)


class TestXOR(unittest.TestCase):
    """Acceptance test of issue #4: XOR is not linearly separable, so solving it
    proves forward, backward, loss and optimizer work together."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_xor_adam(self):
        model = xor_model(Adam(lr=0.05))
        history = model.fit(XOR_X, XOR_Y, epochs=300, batch_size=4, val_split=0.0)
        self.assertEqual(model.predict(XOR_X), ["0", "1", "1", "0"])
        self.assertEqual(history["train_acc"][-1], 1.0)
        self.assertLess(history["train_loss"][-1], 0.02)

    def test_xor_sgd_momentum(self):
        model = xor_model(SGD(lr=0.1, momentum=0.9))
        history = model.fit(XOR_X, XOR_Y, epochs=300, batch_size=4, val_split=0.0)
        self.assertEqual(history["train_acc"][-1], 1.0)
        self.assertLess(history["train_loss"][-1], 0.02)


class TestSequential(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        # Two Gaussian blobs: an easy, linearly separable 2-class problem
        self.X = np.vstack([np.random.randn(100, 2) - 2, np.random.randn(100, 2) + 2]).astype(np.float32)
        self.y = np.repeat(np.eye(2, dtype=np.float32), 100, axis=0)

    def test_predict_proba_sums_to_one(self):
        model = xor_model(Adam())
        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (200, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    def test_validation_split_and_history(self):
        model = xor_model(Adam(lr=0.01))
        history = model.fit(self.X, self.y, epochs=5, batch_size=32, val_split=0.2, patience=None)
        for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            self.assertEqual(len(history[key]), 5)
        self.assertGreater(history["val_acc"][-1], 0.9)
        self.assertIsNotNone(history["best_epoch"])

    def test_early_stopping_restores_best_weights(self):
        # Validation labels are the opposite of the training labels: val loss gets worse
        # as training improves, so early stopping must trigger and roll back.
        model = xor_model(Adam(lr=0.05))
        history = model.fit(self.X, self.y, epochs=50, batch_size=32, patience=3,
                            validation_data=(self.X, self.y[:, ::-1].copy()))
        self.assertLess(len(history["train_loss"]), 50)
        best = history["best_epoch"]
        self.assertEqual(len(history["train_loss"]), best + 3)
        val_loss, _ = model.evaluate(self.X, self.y[:, ::-1].copy())
        self.assertAlmostEqual(val_loss, history["val_loss"][best - 1], places=5)

    def test_reproducible_with_seed(self):
        losses = []
        for _ in range(2):
            np.random.seed(123)
            model = Sequential([Linear(2, 8), ReLU(), Dropout(0.3), Linear(8, 2)],
                               SoftmaxCrossEntropyLoss(), Adam(lr=0.01), XOR_CLASSES)
            losses.append(model.fit(self.X, self.y, epochs=3, batch_size=16)["train_loss"])
        self.assertEqual(losses[0], losses[1])

    def test_fit_leaves_eval_mode(self):
        model = Sequential([Linear(2, 4), Dropout(0.5), Linear(4, 2)], SoftmaxCrossEntropyLoss(), Adam(), XOR_CLASSES)
        model.fit(self.X, self.y, epochs=1, batch_size=32, val_split=0.0)
        self.assertTrue(all(not layer.training for layer in model.layers))
        np.testing.assert_array_equal(model.predict_proba(self.X), model.predict_proba(self.X))

    def test_mse_regression(self):
        X = np.random.rand(64, 1).astype(np.float32)
        y = 3 * X + 1
        model = Sequential([Linear(1, 1, init="xavier")], MSELoss(), SGD(lr=0.1), ["y"])
        history = model.fit(X, y, epochs=200, batch_size=16, val_split=0.0)
        self.assertLess(history["train_loss"][-1], 1e-3)

    def test_config_and_errors(self):
        model = xor_model(Adam())
        config = model.get_config()
        self.assertEqual(config["loss"], "cross_entropy")
        self.assertEqual(config["layers"][1], {"type": "relu"})
        with self.assertRaises(ValueError):
            Sequential([], SoftmaxCrossEntropyLoss(), Adam(), XOR_CLASSES)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y[:10], epochs=1, batch_size=4)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y, epochs=0, batch_size=4)
        wrong_classes = Sequential([Linear(2, 3)], SoftmaxCrossEntropyLoss(), Adam(), XOR_CLASSES)
        with self.assertRaises(ValueError):
            wrong_classes.predict(self.X)


if __name__ == "__main__":
    unittest.main()
