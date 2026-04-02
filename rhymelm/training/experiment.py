"""Experiment tracking: log hyperparameters, metrics, and samples per run."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """JSON-based experiment logger. Each run gets a unique directory."""

    def __init__(self, experiment_dir: str, config: Any = None):
        self.dir = Path(experiment_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.samples_path = self.dir / "samples.jsonl"
        self.config_path = self.dir / "config.json"

        if config is not None:
            self._save_config(config)

        self.start_time = time.time()

    def _save_config(self, config):
        try:
            data = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
        except TypeError:
            data = str(config)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def log_metrics(self, step: int, **kwargs):
        entry = {"step": step, "time": time.time() - self.start_time, **kwargs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_sample(self, step: int, text: str, metadata: dict | None = None):
        entry = {"step": step, "text": text, **(metadata or {})}
        with open(self.samples_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_metrics(self) -> list[dict]:
        if not self.metrics_path.exists():
            return []
        with open(self.metrics_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_samples(self) -> list[dict]:
        if not self.samples_path.exists():
            return []
        with open(self.samples_path) as f:
            return [json.loads(line) for line in f if line.strip()]


def compare_experiments(experiment_dirs: list[str]) -> dict[str, dict]:
    """Load and compare metrics across multiple experiment runs."""
    results = {}
    for d in experiment_dirs:
        logger = ExperimentLogger(d)
        metrics = logger.load_metrics()
        config_path = Path(d) / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        results[Path(d).name] = {
            "config": config,
            "metrics": metrics,
            "final_metrics": metrics[-1] if metrics else {},
        }
    return results
