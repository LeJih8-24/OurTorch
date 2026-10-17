import unittest
import numpy as np

from mytorch.initializers import (
    get_initializer,
    he_normal,
    lecun_normal,
    xavier_normal,
    xavier_uniform,
)
from mytorch.layers.base import Layer


class DummyLayer(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class TestInitializersAndBaseLayer(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.shape = (500, 300)
        self.n_in, self.n_out = self.shape

    def test_xavier_normal_shape_dtype_variance(self):
        weights = xavier_normal(self.shape)
        self.assertEqual(weights.shape, self.shape)
        self.assertEqual(weights.dtype, np.float32)
        expected_var = 2.0 / (self.n_in + self.n_out)
        self.assertAlmostEqual(float(np.mean(weights)), 0.0, delta=0.01)
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.1)

    def test_xavier_uniform_shape_dtype_bounds_variance(self):
        weights = xavier_uniform(self.shape)
        self.assertEqual(weights.shape, self.shape)
        self.assertEqual(weights.dtype, np.float32)
        limit = np.sqrt(6.0 / (self.n_in + self.n_out))
        expected_var = 2.0 / (self.n_in + self.n_out)
        self.assertTrue(np.all(weights >= -limit))
        self.assertTrue(np.all(weights <= limit))
        self.assertAlmostEqual(float(np.mean(weights)), 0.0, delta=0.01)
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.1)

    def test_he_normal_shape_dtype_variance(self):
        weights = he_normal(self.shape)
        self.assertEqual(weights.shape, self.shape)
        self.assertEqual(weights.dtype, np.float32)
        expected_var = 2.0 / self.n_in
        self.assertAlmostEqual(float(np.mean(weights)), 0.0, delta=0.01)
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.1)

    def test_lecun_normal_shape_dtype_variance(self):
        weights = lecun_normal(self.shape)
        self.assertEqual(weights.shape, self.shape)
        self.assertEqual(weights.dtype, np.float32)
        expected_var = 1.0 / self.n_in
        self.assertAlmostEqual(float(np.mean(weights)), 0.0, delta=0.01)
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.1)

    def test_get_initializer_valid(self):
        self.assertIs(get_initializer("xavier_normal"), xavier_normal)
        self.assertIs(get_initializer("xavier_uniform"), xavier_uniform)
        self.assertIs(get_initializer("he_normal"), he_normal)
        self.assertIs(get_initializer("lecun_normal"), lecun_normal)

    def test_get_initializer_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_initializer("unknown_init")

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            xavier_normal((0, 10))
        with self.assertRaises(ValueError):
            he_normal((10, -5))

    def test_base_layer_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            Layer()  # type: ignore[abstract]

    def test_base_layer_defaults(self):
        layer = DummyLayer()
        self.assertTrue(layer.training)
        self.assertEqual(layer.parameters(), [])
        self.assertEqual(layer.get_config(), {"type": "DummyLayer"})
        x = np.ones((2, 3), dtype=np.float32)
        np.testing.assert_array_equal(layer.forward(x), x)
        np.testing.assert_array_equal(layer.backward(x), x)


if __name__ == "__main__":
    unittest.main()
