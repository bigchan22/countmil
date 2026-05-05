import sys
from pathlib import Path
from torchvision import datasets, transforms

root = Path("data")
root.mkdir(parents=True, exist_ok=True)

tfm = transforms.ToTensor()

name = sys.argv[1].lower()

print(f"\n=== Downloading {name} ===", flush=True)

if name == "mnist":
    datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
    datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)

elif name == "fashion":
    datasets.FashionMNIST(root=str(root), train=True, download=True, transform=tfm)
    datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tfm)

elif name == "cifar10":
    datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm)
    datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)

elif name == "cifar100":
    datasets.CIFAR100(root=str(root), train=True, download=True, transform=tfm)
    datasets.CIFAR100(root=str(root), train=False, download=True, transform=tfm)

elif name == "svhn":
    datasets.SVHN(root=str(root), split="train", download=True, transform=tfm)
    datasets.SVHN(root=str(root), split="test", download=True, transform=tfm)

elif name == "stl10":
    datasets.STL10(root=str(root), split="train", download=True, transform=tfm)
    datasets.STL10(root=str(root), split="test", download=True, transform=tfm)

else:
    raise ValueError(f"Unknown dataset: {name}")

print(f"=== Finished {name} ===", flush=True)
