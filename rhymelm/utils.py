"""Utilities: device selection (including ROCm), seeding, parameter helpers."""

import os
import random
import torch
import numpy as np


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if rocm:
            print(f"Using ROCm: {name} (HIP {torch.version.hip})")
        else:
            print(f"Using CUDA: {name}")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon (MPS)")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_param_groups(model: torch.nn.Module, weight_decay: float = 0.01):
    """Separate weight decay: apply only to weight matrices, not biases or norms."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name or "ln" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
