import unittest

import torch

from countmil.models import AttentionMILMNIST, MNISTDigitClassifier, ShuklaMNISTSelector


class MNISTModelTests(unittest.TestCase):
    def test_selector_accepts_bag_tensor(self):
        model = ShuklaMNISTSelector(output_mode="logits")
        x = torch.randn(3, 7, 1, 28, 28)
        y = model(x)
        self.assertEqual(tuple(y.shape), (3, 7))

    def test_selector_accepts_instance_tensor(self):
        model = ShuklaMNISTSelector(output_mode="log_probs")
        x = torch.randn(5, 1, 28, 28)
        y = model(x)
        self.assertEqual(tuple(y.shape), (5,))
        self.assertTrue((y <= 0).all())

    def test_predict_proba_range(self):
        model = ShuklaMNISTSelector(output_mode="logits")
        x = torch.randn(2, 4, 1, 28, 28)
        p = model.predict_proba(x)
        self.assertEqual(tuple(p.shape), (2, 4))
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_gradients_flow(self):
        model = ShuklaMNISTSelector()
        x = torch.randn(2, 3, 1, 28, 28)
        loss = model(x).mean()
        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_attention_mil_shapes_and_masking(self):
        model = AttentionMILMNIST(gated=False)
        x = torch.randn(2, 5, 1, 28, 28)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        logits, weights = model(x, mask)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(weights.shape), (2, 5))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue((weights[0, 3:] == 0).all())

    def test_gated_attention_mil_gradients_flow(self):
        model = AttentionMILMNIST(gated=True)
        x = torch.randn(2, 4, 1, 28, 28)
        logits, weights = model(x)
        loss = logits.mean() + weights.mean()
        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_digit_classifier_accepts_bag_tensor(self):
        model = MNISTDigitClassifier()
        x = torch.randn(2, 3, 1, 28, 28)
        logits = model(x)
        probs = model.predict_proba(x)
        self.assertEqual(tuple(logits.shape), (2, 3, 10))
        self.assertEqual(tuple(probs.shape), (2, 3, 10))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2, 3), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
