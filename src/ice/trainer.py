from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ice.utils import find_device


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 0.01
    epochs: int = 10
    batch_size: int = 32

    # advanced params
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0
    nesterov: bool = False

    # likely won't need to touch those
    split_size: float = 0.7
    seed: int | None = None
    device: str = "auto"


class Trainer:
    """Trainer class, inspired by HuggingFace's Trainer and PyTorch Lightning's Trainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        config: TrainerConfig,
        dataset: Dataset,
    ):
        # core attributes
        self.config = config

        # direct access attributes
        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.device = find_device(self.config.device)

        self.model = model

        # important! Otherwise the optimiser will not update the correct weights
        self.model.to(self.device)

        self.optimizer = self.configure_optimizer()
        self.loss_fn = self.configure_loss()

        self.train_dataloader, self.val_dataloader = self.configure_dataloaders(dataset)

    def __repr__(self) -> str:
        return f"Trainer(\nmodel={self.model},\nconfig={self.config},\nOptimizer={self.optimizer})"

    def configure_dataloaders(
        self,
        dataset: Dataset,
    ) -> tuple[DataLoader, DataLoader]:
        if not 0 < (split_size := self.config.split_size) < 1:
            raise ValueError("Split size must be between 0 and 1")

        if (seed := self.config.seed) is not None:
            self.rng = torch.Generator().manual_seed(seed)
        else:
            self.rng = None

        train_subset, val_subset = random_split(
            dataset,
            (split_size, 1 - split_size),
            generator=self.rng,
        )

        return (
            DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
            ),
            DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=True,
            ),
        )

    def configure_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            dampening=self.config.dampening,
            nesterov=self.config.nesterov,
        )

    def configure_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def training_step(self, X, y) -> torch.Tensor:
        # Move data to device
        X, y = X.to(self.device, dtype=torch.float16), y.to(self.device)

        # Forward pass
        logits = self.model(X)
        return self.loss_fn(logits, y)

    def validation_step(self, X, y) -> tuple[torch.Tensor, torch.Tensor]:
        # Move data to device
        X, y = X.to(self.device, dtype=torch.float16), y.to(self.device)

        # Forward pass
        logits = self.model(X)
        return logits, self.loss_fn(logits, y)

    def training_loop(self) -> None:
        size = len(self.train_dataloader.dataset)

        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()

        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            loss = self.training_step(X, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * self.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def validation_loop(self) -> None:
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        test_loss, correct = 0, 0

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in self.val_dataloader:
                predictions, loss = self.validation_step(X, y)
                test_loss += loss.item()
                correct += (predictions.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def fit(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.training_loop()
            self.validation_loop()
        print("Done!")
