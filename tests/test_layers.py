import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.layers import LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Tanh
from tests.gradcheck import check_layer_gradients

GRAD_TOLERANCE = 1e-5


def linear_float64(in_features: int, out_features: int) -> Linear:
    """Linear layer whose parameters (and gradients) are cast to float64 for gradient checks."""
    layer = Linear(in_features, out_features)
    layer.W = layer.W.astype(np.float64)
    layer.b = np.random.standard_normal(layer.b.shape)
    layer.dW = np.zeros_like(layer.W)
    layer.db = np.zeros_like(layer.b)
    return layer


class TestLinear(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_shapes_and_init(self):
        layer = Linear(4, 3)
        self.assertEqual(layer.W.shape, (4, 3))
        self.assertEqual(layer.b.shape, (1, 3))
        self.assertEqual(layer.W.dtype, np.float32)
        np.testing.assert_array_equal(layer.b, 0.0)
        out = layer.forward(np.ones((5, 4), dtype=np.float32))
        self.assertEqual(out.shape, (5, 3))

    def test_forward_value(self):
        layer = Linear(2, 2)
        layer.W[...] = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.b[...] = np.array([[0.5, -0.5]])
        out = layer.forward(np.array([[1.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(out, [[4.5, 5.5]])

    def test_gradient_check(self):
        layer = linear_float64(5, 4)
        X = np.random.standard_normal((6, 5))
        for error in check_layer_gradients(layer, X):
            self.assertLess(error, GRAD_TOLERANCE)

    def test_parameters_references_are_stable(self):
        layer = Linear(3, 2)
        params_before = layer.parameters()
        layer.forward(np.ones((4, 3), dtype=np.float32))
        layer.backward(np.ones((4, 2), dtype=np.float32))
        for (p0, g0), (p1, g1) in zip(params_before, layer.parameters()):
            self.assertIs(p0, p1)
            self.assertIs(g0, g1)
        np.testing.assert_allclose(layer.db, [[4.0, 4.0]])

    def test_initializer_choice(self):
        layer = Linear(10, 5, init="xavier")
        self.assertEqual(layer.get_config(), {"type": "linear", "in": 10, "out": 5, "init": "xavier"})
        with self.assertRaises(ValueError):
            Linear(10, 5, init="unknown")

    def test_invalid_usage_raises(self):
        with self.assertRaises(ValueError):
            Linear(0, 5)
        layer = Linear(3, 2)
        with self.assertRaises(ValueError):
            layer.forward(np.ones((4, 5)))
        with self.assertRaises(RuntimeError):
            Linear(3, 2).backward(np.ones((4, 2)))


def inputs_away_from_zero(shape, seed: int = 1) -> np.ndarray:
    """Random float64 inputs with |x| >= 0.1, so finite differences never cross the ReLU kink."""
    X = np.random.default_rng(seed).standard_normal(shape)
    return np.where(X >= 0, X + 0.1, X - 0.1)


class TestActivations(unittest.TestCase):
    def test_gradient_checks(self):
        X = inputs_away_from_zero((4, 6))
        for layer in (ReLU(), LeakyReLU(0.1), Sigmoid(), Tanh(), Softmax()):
            with self.subTest(layer=layer.__class__.__name__):
                for error in check_layer_gradients(layer, X.copy()):
                    self.assertLess(error, GRAD_TOLERANCE)

    def test_forward_values(self):
        x = np.array([[-2.0, 0.0, 3.0]])
        np.testing.assert_allclose(ReLU().forward(x), [[0.0, 0.0, 3.0]])
        np.testing.assert_allclose(LeakyReLU(0.1).forward(x), [[-0.2, 0.0, 3.0]])
        np.testing.assert_allclose(Sigmoid().forward(np.zeros((1, 2))), [[0.5, 0.5]])
        np.testing.assert_allclose(Tanh().forward(x), np.tanh(x))

    def test_sigmoid_no_overflow(self):
        with np.errstate(over="raise"):
            out = Sigmoid().forward(np.array([[-1000.0, 1000.0]]))
        np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-12)

    def test_softmax_stable_and_normalized(self):
        out = Softmax().forward(np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]]))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])
        np.testing.assert_allclose(out[1], [1 / 3] * 3)

    def test_configs(self):
        self.assertEqual(ReLU().get_config(), {"type": "relu"})
        self.assertEqual(LeakyReLU(0.2).get_config(), {"type": "leakyrelu", "alpha": 0.2})
        self.assertEqual(Softmax().get_config(), {"type": "softmax"})
        self.assertEqual(ReLU().parameters(), [])

    def test_backward_before_forward_raises(self):
        for layer in (ReLU(), LeakyReLU(), Sigmoid(), Tanh(), Softmax()):
            with self.assertRaises(RuntimeError):
                layer.backward(np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
