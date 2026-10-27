import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.optimizers import SGD, Adam, get_optimizer


def minimize_quadratic(optimizer, steps: int = 500) -> np.ndarray:
    """Minimize f(p) = ||p - target||^2 (gradient 2 * (p - target)) and return p."""
    target = np.array([3.0, -2.0])
    param = np.zeros(2)
    grad = np.zeros(2)
    for _ in range(steps):
        grad[...] = 2.0 * (param - target)
        optimizer.step([(param, grad)])
    return param


class TestSGD(unittest.TestCase):
    def test_plain_step(self):
        param, grad = np.array([1.0, 2.0]), np.array([0.5, -1.0])
        SGD(lr=0.1, momentum=0.0).step([(param, grad)])
        np.testing.assert_allclose(param, [0.95, 2.1])

    def test_momentum_accumulates(self):
        param, grad = np.array([0.0]), np.array([1.0])
        opt = SGD(lr=0.1, momentum=0.9)
        opt.step([(param, grad)])
        np.testing.assert_allclose(param, [-0.1])
        opt.step([(param, grad)])
        np.testing.assert_allclose(param, [-0.1 - 0.19])

    def test_converges(self):
        for momentum in (0.0, 0.9):
            with self.subTest(momentum=momentum):
                np.testing.assert_allclose(minimize_quadratic(SGD(lr=0.05, momentum=momentum)), [3.0, -2.0], atol=1e-4)

    def test_weight_decay_shrinks_weights(self):
        param, grad = np.array([10.0]), np.array([0.0])
        SGD(lr=0.1, momentum=0.0, weight_decay=0.5).step([(param, grad)])
        np.testing.assert_allclose(param, [9.5])


class TestAdam(unittest.TestCase):
    def test_first_step_magnitude_is_lr(self):
        # With bias correction, the first Adam step is ~lr * sign(grad)
        param, grad = np.array([1.0, 1.0]), np.array([0.001, -50.0])
        Adam(lr=0.01).step([(param, grad)])
        np.testing.assert_allclose(param, [0.99, 1.01], atol=1e-6)

    def test_converges(self):
        np.testing.assert_allclose(minimize_quadratic(Adam(lr=0.05), steps=2000), [3.0, -2.0], atol=1e-3)

    def test_state_is_per_parameter(self):
        p1, g1 = np.array([0.0]), np.array([1.0])
        p2, g2 = np.array([0.0]), np.array([-1.0])
        Adam(lr=0.1).step([(p1, g1), (p2, g2)])
        np.testing.assert_allclose(p1, -p2)


class TestOptimizerFactory(unittest.TestCase):
    def test_get_optimizer(self):
        opt = get_optimizer({"type": "adam", "lr": 0.002, "weight_decay": 1e-4})
        self.assertIsInstance(opt, Adam)
        self.assertEqual(opt.lr, 0.002)
        self.assertEqual(get_optimizer({"type": "SGD"}).get_config()["momentum"], 0.9)

    def test_invalid_configs(self):
        for config in ({"lr": 0.1}, {"type": "rmsprop"}, {"type": "sgd", "foo": 1},
                       {"type": "adam", "lr": -1}, {"type": "sgd", "momentum": 1.0}, "adam"):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    get_optimizer(config)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
