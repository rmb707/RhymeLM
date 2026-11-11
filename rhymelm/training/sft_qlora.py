"""QLoRA fine-tune of an open base LLM on the private lyric corpus.

Reads data/private/corpus/{train,val}.jsonl (built by rhymelm.ingest.build_corpus),
trains a LoRA adapter on a 4-bit base model, and writes:
    checkpoints/<run>/adapter/         LoRA weights (gitignored; trained on private data)
    reports/runs/<run>/metrics.json    losses + config + env (no text; safe to commit)

Written against transformers 5.22, trl 1.17, peft 0.21, bitsandbytes 0.50.

Usage:
    python -m rhymelm.training.sft_qlora --config configs/qlora_qwen3_8b.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import yaml

# Kept stable across training and generation: the prompt format IS the model's API.
KIND_TAG = {"lyrics": "verse", "fragment": "bars", "prose": "piece"}


def format_doc(doc: dict, eos: str) -> str:
    title = (doc.get("title") or "").strip() or "Untitled"
    kind = KIND_TAG.get(doc.get("kind", "lyrics"), "verse")
    return f"Title: {title}\nType: {kind}\n\n{doc['text'].strip()}\n{eos}"


def generation_prompt(title: str, kind: str = "verse") -> str:
    return f"Title: {title.strip()}\nType: {kind}\n\n"


def load_split(path: Path, eos: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [{"text": format_doc(r, eos)} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    t = cfg["train"]
    set_seed(t["seed"])
    run = cfg["run_name"] + time.strftime("-%Y%m%dT%H%M%SZ", time.gmtime())
    ckpt_dir = Path("checkpoints") / run
    report_dir = Path("reports/runs") / run
    report_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    corpus = Path(cfg["corpus_dir"])
    train_ds = Dataset.from_list(load_split(corpus / "train.jsonl", tok.eos_token))
    val_ds = Dataset.from_list(load_split(corpus / "val.jsonl", tok.eos_token))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        revision=cfg.get("revision"),
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )

    l = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        use_rslora=l.get("rslora", False),
    )

    sft = SFTConfig(
        output_dir=str(ckpt_dir),
        max_length=t["max_length"],
        packing=t["packing"],
        num_train_epochs=t["epochs"],
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        lr_scheduler_type=t["scheduler"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        neftune_noise_alpha=t.get("neftune_noise_alpha"),
        gradient_checkpointing=True,
        bf16=True,
        optim=t.get("optim", "paged_adamw_8bit"),
        eval_strategy="epoch" if not args.max_steps else "no",
        save_strategy="epoch" if not args.max_steps else "no",
        save_total_limit=2,
        load_best_model_at_end=not args.max_steps,
        metric_for_best_model="eval_loss",
        logging_steps=5,
        report_to="none",
        seed=t["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    final_eval = trainer.evaluate() if len(val_ds) else {}

    adapter_dir = ckpt_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    metrics = {
        "run": run,
        "config": cfg,
        "train_docs": len(train_ds),
        "val_docs": len(val_ds),
        "trainable_params": trainable,
        "train_seconds": round(elapsed, 1),
        "final_eval": final_eval,
        "log_history": trainer.state.log_history,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": __import__("transformers").__version__,
            "trl": __import__("trl").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
        },
        "adapter_dir": str(adapter_dir),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"done: {run} eval={final_eval} -> {report_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
