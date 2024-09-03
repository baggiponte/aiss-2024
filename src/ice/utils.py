from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as tfs
from ipywidgets import interact
from torchvision.datasets import MNIST

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Dataset


def load_mnist(
    path: str = "./data",
    transforms: Callable | None = None,
) -> tuple[MNIST, MNIST]:
    if transforms is None:
        transforms = tfs.Compose([tfs.ToImage()])

    train_set = MNIST(path, train=True, download=True, transform=transforms)
    test_set = MNIST(path, train=False, download=True, transform=transforms)
    return train_set, test_set


def display(dataset: Dataset, **kwargs):
    size = len(dataset)

    @interact(index=(1, size, 1))
    def inner(index):
        picture, label = dataset[index - 1]

        cmap = kwargs.pop("cmap", "gray")

        plt.imshow(picture.permute(1, 2, 0), cmap=cmap, **kwargs)
        plt.title(dataset.classes[label])


def find_device(device):
    if device == "auto" and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            raise OSError(
                "PyTorch not built with MPS enabled. You cannot use the 'mps' device."
            )
        return "mps"
    elif device == "auto" and torch.cuda.is_available():
        return "cuda"
    elif device == "auto" and not torch.cuda.is_available():
        return "cpu"
    elif device not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError("Devices can only be ('auto', 'cuda', 'mps', 'cpu')")
    else:
        return device
