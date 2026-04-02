"""Dataclass-based configuration for all RhymeLM components."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    csv_path: str = "lyrics_raw.csv"
    lyrics_column: str = "artist_verses"
    min_bars: int = 8
    max_bars: int = 16
    dict_ratio: float = 0.25
    dict_words_per_block: int = 50
    val_split: float = 0.1


@dataclass
class ModelConfig:
    arch: str = "lstm"
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 64
    block_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_steps: int = 50_000
    warmup_steps: int = 2_500
    eval_interval: int = 1_000
    sample_interval: int = 5_000
    checkpoint_interval: int = 10_000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    label_smoothing: float = 0.0
    seed: int = 42


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.0
    repetition_penalty: float = 1.0
    num_bars: int = 16
    max_chars: int = 2000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "rhymelm"

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            checkpoint_dir=raw.get("checkpoint_dir", "checkpoints"),
            experiment_name=raw.get("experiment_name", "rhymelm"),
        )
