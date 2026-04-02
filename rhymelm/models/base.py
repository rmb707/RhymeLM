"""Abstract base class for all RhymeLM model variants."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RhymeLMBase(ABC, nn.Module):
    """Base class defining the interface all RhymeLM models must implement."""

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        """Forward pass returning (logits, state)."""
        ...

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize any recurrent state for generation."""
        ...

    def _init_weights(self):
        """Xavier initialization for better training dynamics."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
