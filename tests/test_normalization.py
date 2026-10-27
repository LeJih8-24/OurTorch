import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.layers import Linear, Normalization, ReLU
from tests.gradcheck import check_layer_gradients

GRAD_TOLERANCE = 1e-5


class TestNormalization(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        # Features with very different scales, plus one constant column
        self.X = np.column_stack([
            rng.normal(100.0, 20.0, 500),
            rng.normal(-3.0, 0.01, 500),
            rng.integers(0, 2, 500),
            np.full(500, 7.0),
        ]).astype(np.float32)

    def test_identity_before_adapt(self):
        layer = Normalization(4)
        self.assertFalse(layer.adapted)
        np.testing.assert_allclose(layer.forward(self.X), self.X)

    def test_adapt_standardizes(self):
        layer = Normalization(4)
        layer.adapt(self.X)
        self.assertTrue(layer.adapted)
        self.assertEqual(int(layer.count[0]), 500)
        out = layer.forward(self.X)
        np.testing.assert_allclose(out[:, :3].mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(out[:, :3].std(axis=0), 1.0, atol=1e-3)

    def test_constant_feature_is_only_centered(self):
        layer = Normalization(4)
        layer.adapt(self.X)
        self.assertEqual(float(layer.std[0, 3]), 1.0)
        out = layer.forward(self.X)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[:, 3], 0.0)

    def test_not_trainable_and_config(self):
        layer = Normalization(4)
        self.assertEqual(layer.parameters(), [])
        self.assertEqual(layer.get_config(), {"type": "normalization", "features": 4})

    def test_gradient_check(self):
        layer = Normalization(4)
        layer.adapt(self.X.astype(np.float64))
        layer.mean = layer.mean.astype(np.float64)
        layer.std = layer.std.astype(np.float64)
        X = np.random.default_rng(1).standard_normal((5, 4))
        for error in check_layer_gradients(layer, X):
            self.assertLess(error, GRAD_TOLERANCE)

    def test_errors(self):
        with self.assertRaises(ValueError):
            Normalization(0)
        layer = Normalization(4)
        with self.assertRaises(ValueError):
            layer.forward(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            layer.adapt(np.ones((0, 4)))
        with self.assertRaises(RuntimeError):
            Normalization(4).backward(np.ones((2, 4)))


class TestStateDict(unittest.TestCase):
    def test_linear_round_trip(self):
        np.random.seed(0)
        source, target = Linear(3, 2), Linear(3, 2)
        W_ref = target.W
        target.load_state_dict({k: v.copy() for k, v in source.state_dict().items()})
        np.testing.assert_array_equal(target.W, source.W)
        self.assertIs(target.W, W_ref)  # updated in place

    def test_normalization_round_trip_keeps_adapted_flag(self):
        source = Normalization(2)
        source.adapt(np.array([[0.0, 10.0], [2.0, 30.0]]))
        target = Normalization(2)
        target.load_state_dict(source.state_dict())
        self.assertTrue(target.adapted)
        np.testing.assert_allclose(target.mean, [[1.0, 20.0]])
        blank = Normalization(2)
        blank.load_state_dict(Normalization(2).state_dict())
        self.assertFalse(blank.adapted)

    def test_stateless_and_invalid_states(self):
        self.assertEqual(ReLU().state_dict(), {})
        layer = Linear(3, 2)
        with self.assertRaises(ValueError):
            layer.load_state_dict({"W": np.zeros((3, 2))})
        with self.assertRaises(ValueError):
            layer.load_state_dict({"W": np.zeros((2, 2)), "b": np.zeros((1, 2))})


if __name__ == "__main__":
    unittest.main()
