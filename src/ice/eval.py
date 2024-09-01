from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ipywidgets import interact

from ice.utils import find_device

if TYPE_CHECKING:
    from typing import Literal

    from torch.nn import Module
    from torch.utils.data import Dataset


def predict(
    *,
    model: Module,
    dataset: Dataset,
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
):
    device = find_device(device)

    size = len(dataset)

    @interact(index=(1, size, 1))
    def inner(index):
        idx = index - 1
        model.eval()
        with torch.no_grad():
            row = dataset.data[idx].to(device=device, dtype=torch.float32).view(1, -1)

            logits = model(row)
            prediction = logits.argmax()

        truth = dataset.targets[idx]
        predicted_label = dataset.classes[prediction]
        true_label = dataset.classes[truth]

        print("predicted label:\t", predicted_label, "\ntrue label:\t\t", true_label)


def evaluate(
    *,
    model: Module,
    dataset: Dataset,
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
):
    device = find_device(device)
    with torch.no_grad():
        data = dataset.data.to(device, dtype=torch.float32)
        logits = model(data)
        predictions = logits.argmax(1)

    size = len(dataset)
    percentage = (predictions == dataset.targets.to(device)).sum() / size
    return percentage.item()
