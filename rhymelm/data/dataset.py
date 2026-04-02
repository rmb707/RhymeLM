"""PyTorch Dataset and DataLoader wrappers for language model training."""

import torch
from torch.utils.data import Dataset, DataLoader


class LMDataset(Dataset):
    """Character-level language modeling dataset.

    Each sample is a (x, y) pair where y is x shifted by one position.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_dataloaders(
    encoded: torch.Tensor,
    block_size: int,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from encoded corpus."""
    split_idx = int(len(encoded) * (1 - val_split))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    train_ds = LMDataset(train_data, block_size)
    val_ds = LMDataset(val_data, block_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    print(f"Train: {len(train_data):,} chars ({len(train_ds):,} samples)")
    print(f"Val:   {len(val_data):,} chars ({len(val_ds):,} samples)")
    return train_loader, val_loader
