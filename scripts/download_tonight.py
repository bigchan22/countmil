from pathlib import Path
from torchvision import datasets, transforms

root = Path("data")
root.mkdir(parents=True, exist_ok=True)

tfm = transforms.ToTensor()

def download(name, fn):
    print(f"\n=== Downloading {name} ===", flush=True)
    fn()
    print(f"=== Finished {name} ===", flush=True)

download("MNIST", lambda: (
    datasets.MNIST(root=str(root), train=True, download=True, transform=tfm),
    datasets.MNIST(root=str(root), train=False, download=True, transform=tfm),
))

download("FashionMNIST", lambda: (
    datasets.FashionMNIST(root=str(root), train=True, download=True, transform=tfm),
    datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tfm),
))

download("CIFAR10", lambda: (
    datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm),
    datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm),
))

download("CIFAR100", lambda: (
    datasets.CIFAR100(root=str(root), train=True, download=True, transform=tfm),
    datasets.CIFAR100(root=str(root), train=False, download=True, transform=tfm),
))

download("SVHN train/test", lambda: (
    datasets.SVHN(root=str(root), split="train", download=True, transform=tfm),
    datasets.SVHN(root=str(root), split="test", download=True, transform=tfm),
))

download("STL10 train/test", lambda: (
    datasets.STL10(root=str(root), split="train", download=True, transform=tfm),
    datasets.STL10(root=str(root), split="test", download=True, transform=tfm),
))

print("\nAll basic datasets downloaded.")
print(f"Data root: {root.resolve()}")
