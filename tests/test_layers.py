import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.layers import Linear
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


if __name__ == "__main__":
    unittest.main()
