"""DPO on the SFT adapter: teach the model to prefer its own best-rhyming verses.

Loads base (4-bit) + the SFT LoRA adapter as TRAINABLE. TRL 1.17 copies that
adapter into a frozen "ref" adapter, so the KL anchor is the SFT model (Rob's
voice), not the untuned base. Pairs come from rhymelm.training.dpo_data.

Writes checkpoints/<run>/adapter (private) and reports/runs/<run>/metrics.json.
Written against trl 1.17.1, peft 0.21.2, transformers 5.22.2.

Usage:
    python -m rhymelm.training.dpo_train --config configs/dpo_v3.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    t = cfg["train"]
    set_seed(t["seed"])
    run = cfg["run_name"] + time.strftime("-%Y%m%dT%H%M%SZ", time.gmtime())
    ckpt_dir = Path("checkpoints") / run
    report_dir = Path("reports/runs") / run
    report_dir.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in open(cfg["pairs"], encoding="utf-8")]
    rng = random.Random(t["seed"])
    rng.shuffle(pairs)
    n_eval = max(4, int(len(pairs) * cfg.get("eval_fraction", 0.1)))
    cols = ("prompt", "chosen", "rejected")
    to_ds = lambda rows: Dataset.from_list([{k: r[k] for k in cols} for r in rows])
    eval_ds, train_ds = to_ds(pairs[:n_eval]), to_ds(pairs[n_eval:])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg.get("revision"),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, cfg["sft_adapter"], is_trainable=True)

    dpo = DPOConfig(
        output_dir=str(ckpt_dir),
        beta=t["beta"],
        loss_type=t.get("loss_type", ["sigmoid"]),
        max_length=t["max_length"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        lr_scheduler_type=t["scheduler"],
        warmup_steps=t["warmup_steps"],
        gradient_checkpointing=True,
        bf16=True,
        optim=t.get("optim", "paged_adamw_8bit"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=2,
        report_to="none",
        seed=t["seed"],
    )
    trainer = DPOTrainer(model=model, args=dpo, train_dataset=train_ds,
                         eval_dataset=eval_ds, processing_class=tok)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    final_eval = trainer.evaluate()

    adapter_dir = ckpt_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, selected_adapters=["default"])
    tok.save_pretrained(adapter_dir)

    metrics = {
        "run": run, "config": cfg, "train_pairs": len(train_ds), "eval_pairs": len(eval_ds),
        "train_seconds": round(elapsed, 1), "final_eval": final_eval,
        "log_history": trainer.state.log_history,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "trl": __import__("trl").__version__, "peft": __import__("peft").__version__},
        "adapter_dir": str(adapter_dir),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"done: {run} eval={final_eval}")


if __name__ == "__main__":
    main()
