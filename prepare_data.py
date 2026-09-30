"""Explicitly download CIFAR data required by the original offline trainer."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    a = p.parse_args()
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[a.dataset]
    data = Path(__file__).resolve().parent / "data"
    cls(data, train=True, download=True)
    print(f"Prepared {a.dataset} training split at {data}")


if __name__ == "__main__":
    main()
