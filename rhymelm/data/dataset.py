"""PyTorch Dataset and DataLoader wrappers for language model training."""

import torch
from torch.utils.data import Dataset, DataLoader


class LMDataset(Dataset):
    """Language modeling dataset (works with both char and BPE tokenization).

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


class ArtistLMDataset(Dataset):
    """Language modeling dataset with artist conditioning.

    Each verse is encoded separately with its artist ID. Samples return
    (x, y, artist_id) triples so the model can learn artist-specific patterns.
    """

    def __init__(self, verses_encoded: list[tuple[torch.Tensor, int]], block_size: int):
        """
        Args:
            verses_encoded: list of (encoded_verse_tensor, artist_id) pairs
            block_size: context window size
        """
        self.block_size = block_size
        self.samples = []

        for data, artist_id in verses_encoded:
            for i in range(0, len(data) - block_size - 1, block_size // 2):
                x = data[i : i + block_size]
                y = data[i + 1 : i + block_size + 1]
                if len(x) == block_size and len(y) == block_size:
                    self.samples.append((x, y, artist_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, artist_id = self.samples[idx]
        return x, y, torch.tensor(artist_id, dtype=torch.long)


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
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )

    print(f"Train: {len(train_data):,} tokens ({len(train_ds):,} samples)")
    print(f"Val:   {len(val_data):,} tokens ({len(val_ds):,} samples)")
    return train_loader, val_loader


def create_artist_dataloaders(
    verses_encoded: list[tuple[torch.Tensor, int]],
    block_size: int,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create artist-conditioned DataLoaders from per-verse encoded data."""
    import random
    random.shuffle(verses_encoded)
    split = int(len(verses_encoded) * (1 - val_split))
    train_verses = verses_encoded[:split]
    val_verses = verses_encoded[split:]

    train_ds = ArtistLMDataset(train_verses, block_size)
    val_ds = ArtistLMDataset(val_verses, block_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )

    print(f"Artist dataset: {len(train_ds):,} train / {len(val_ds):,} val samples")
    return train_loader, val_loader
