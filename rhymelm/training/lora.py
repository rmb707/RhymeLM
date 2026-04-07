"""LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Instead of updating all 25-30M parameters, LoRA freezes the pretrained
weights and injects small trainable rank-r matrices into attention layers.
At rank 8 with our transformer, this means ~98k trainable params — fine-tuning
in minutes on CPU instead of hours.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter.

    output = frozen_linear(x) + x @ A @ B * scaling

    A is initialized from N(0, 1/r), B is initialized to zeros,
    so the adapter starts as identity (no change to pretrained behavior).
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        d_in = original.in_features
        d_out = original.out_features

        # A projects down to rank, B projects back up
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Freeze model and inject LoRA adapters into target linear layers.

    Args:
        model: the pretrained model
        rank: LoRA rank (lower = fewer params, higher = more expressivity)
        alpha: scaling factor
        target_modules: names of nn.Linear layers to adapt.
            Default: ["qkv_proj", "out_proj"] (attention layers)

    Returns:
        model with LoRA adapters (only adapter params require grad)
    """
    if target_modules is None:
        target_modules = ["qkv_proj", "out_proj"]

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into target modules
    lora_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank, alpha)
                    setattr(module, target, lora_layer)
                    lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {lora_count} layers adapted, {trainable:,} trainable / {total:,} total params ({trainable/total:.1%})")

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into base weights for zero-overhead inference.

    After merging, the model behaves identically but without the adapter
    computation overhead. This is irreversible.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge: W' = W + A @ B * scaling
            with torch.no_grad():
                module.original.weight.add_(
                    (module.lora_A @ module.lora_B).T * module.scaling
                )
                module.original.weight.requires_grad = True
                if module.original.bias is not None:
                    module.original.bias.requires_grad = True

    # Replace LoRALinear modules with their original Linear
    for name, module in model.named_modules():
        for attr_name in list(vars(module)):
            attr = getattr(module, attr_name)
            if isinstance(attr, LoRALinear):
                setattr(module, attr_name, attr.original)

    print("LoRA weights merged into base model")
    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get only the LoRA adapter parameters (for the optimizer)."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
