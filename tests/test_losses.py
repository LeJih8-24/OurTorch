import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from mytorch.losses import MSELoss, SoftmaxCrossEntropyLoss, get_loss, softmax
from tests.gradcheck import check_loss_gradient

GRAD_TOLERANCE = 1e-5


def one_hot(indices, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[indices]


class TestMSELoss(unittest.TestCase):
    def test_value(self):
        loss = MSELoss()
        value = loss.forward(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 0.0], [3.0, 2.0]]))
        self.assertAlmostEqual(value, 2.0)

    def test_gradient_check(self):
        rng = np.random.default_rng(0)
        y_pred, y_true = rng.standard_normal((5, 3)), rng.standard_normal((5, 3))
        self.assertLess(check_loss_gradient(MSELoss(), y_pred, y_true), GRAD_TOLERANCE)


class TestSoftmaxCrossEntropyLoss(unittest.TestCase):
    def test_value_matches_definition(self):
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        y_true = one_hot([0, 1], 3)
        expected = -np.mean(np.log(np.sum(softmax(logits) * y_true, axis=1)))
        self.assertAlmostEqual(SoftmaxCrossEntropyLoss().forward(logits, y_true), expected)

    def test_uniform_logits_give_log_n(self):
        value = SoftmaxCrossEntropyLoss().forward(np.zeros((4, 5)), one_hot([0, 1, 2, 3], 5))
        self.assertAlmostEqual(value, np.log(5))

    def test_stable_with_extreme_logits(self):
        loss = SoftmaxCrossEntropyLoss()
        value = loss.forward(np.array([[1000.0, -1000.0]]), one_hot([1], 2))
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 2000.0)
        self.assertTrue(np.all(np.isfinite(loss.backward())))

    def test_gradient_check(self):
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((6, 4))
        y_true = one_hot(rng.integers(0, 4, size=6), 4)
        self.assertLess(check_loss_gradient(SoftmaxCrossEntropyLoss(), logits, y_true), GRAD_TOLERANCE)

    def test_output_transform_is_softmax(self):
        probs = SoftmaxCrossEntropyLoss().output_transform(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0])


class TestLossFactoryAndErrors(unittest.TestCase):
    def test_get_loss(self):
        self.assertIsInstance(get_loss("mse"), MSELoss)
        self.assertIsInstance(get_loss("Cross_Entropy"), SoftmaxCrossEntropyLoss)
        self.assertEqual(get_loss("cross_entropy").get_config(), {"type": "cross_entropy"})
        with self.assertRaises(ValueError):
            get_loss("hinge")

    def test_shape_mismatch_and_backward_before_forward(self):
        with self.assertRaises(ValueError):
            MSELoss().forward(np.zeros((2, 3)), np.zeros((2, 2)))
        with self.assertRaises(RuntimeError):
            SoftmaxCrossEntropyLoss().backward()


if __name__ == "__main__":
    unittest.main()
