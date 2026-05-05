import unittest

import torch

from countmil.models import make_cifar_classifier


class CIFARModelTests(unittest.TestCase):
    def test_small_cnn_factory_handles_bags(self):
        model = make_cifar_classifier("small_cnn", num_classes=10)
        x = torch.rand(2, 3, 3, 32, 32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 3, 10))

    def test_small_cnn_rejects_pretrained(self):
        with self.assertRaises(ValueError):
            make_cifar_classifier("small_cnn", num_classes=10, pretrained=True)

    def test_resnet18_factory_handles_bags_without_pretrained_download(self):
        model = make_cifar_classifier("resnet18", num_classes=10, pretrained=False)
        x = torch.rand(2, 3, 3, 32, 32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 3, 10))


if __name__ == "__main__":
    unittest.main()
