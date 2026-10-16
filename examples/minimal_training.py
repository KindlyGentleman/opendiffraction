"""Example: Minimal CPU-friendly PyTorch training loop on OpenDiffraction."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from examples.pytorch_dataset import OpenDiffractionDataset


class TinyPhaseRetrievalNet(nn.Module):
    """Minimal CNN baseline reconstructing 64x64 object from 512x512 diffraction pattern."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),  # 256
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 64
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_minimal_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch["input"]
        targets = batch["target"]

        # Normalize target to [0, 1] for demonstration
        t_max = torch.amax(targets, dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        targets_norm = targets / t_max

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


if __name__ == "__main__":
    shard = "output_quick/data/train_000.h5"
    if os.path.exists(shard):
        ds = OpenDiffractionDataset([shard])
        loader = DataLoader(ds, batch_size=4, shuffle=True)

        model = TinyPhaseRetrievalNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print("Running 1 minimal training epoch on CPU...")
        loss = train_minimal_epoch(model, loader, optimizer, criterion)
        print(f"Epoch complete. Average MSE Loss: {loss:.4f}")
