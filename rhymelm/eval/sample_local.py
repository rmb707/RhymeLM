"""Generate eval verses from a local base model, optionally with a LoRA adapter.

Writes reports/samples/<system>.jsonl in the scoreboard format. Uses the same
prompt format the adapter was trained on (rhymelm.training.sft_qlora).

Usage:
    python -m rhymelm.eval.sample_local --system qwen3-8b-base
    python -m rhymelm.eval.sample_local --system rhymelm-v1 --adapter checkpoints/<run>/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhymelm.training.sft_qlora import generation_prompt


def read_prompts(path: str) -> list[str]:
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def clean_verse(text: str, bars: int) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith(("title:", "type:")):
            break  # the model started a new song
        lines.append(line)
        if len(lines) >= bars:
            break
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--revision", default="49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", default="eval/prompts.txt")
    ap.add_argument("--out-dir", default="reports/samples")
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-tokens", type=int, default=None, help="default: 5 x bars")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              StoppingCriteria, StoppingCriteriaList, set_seed)

    class LineStop(StoppingCriteria):
        """Stop once the generated text holds `bars` non-empty lines."""
        def __init__(self, tokenizer, prompt_len: int, bars: int):
            self.tok, self.prompt_len, self.bars = tokenizer, prompt_len, bars

        def __call__(self, input_ids, scores, **kwargs):
            text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            done = sum(1 for l in text.split("\n")[:-1] if l.strip()) >= self.bars
            return torch.tensor([done], device=input_ids.device)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out = Path(args.out_dir) / f"{args.system}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for title in read_prompts(args.prompts):
            ids = tok(generation_prompt(title), return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **ids,
                    max_new_tokens=args.bars * 24,
                    # No EOS before ~half a verse (~8 tokens per half-bar line):
                    # short fragments in training otherwise teach 1-line answers.
                    min_new_tokens=args.min_tokens if args.min_tokens is not None else args.bars * 5,
                    stopping_criteria=StoppingCriteriaList([LineStop(tok, ids["input_ids"].shape[1], args.bars)]),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            text = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            verse = clean_verse(text, args.bars)
            f.write(json.dumps({"system": args.system, "prompt": title, "text": verse,
                                "params": {"temperature": args.temperature, "top_p": args.top_p,
                                           "repetition_penalty": args.repetition_penalty,
                                           "seed": args.seed, "adapter": bool(args.adapter)}},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"{title}: {verse.count(chr(10)) + 1 if verse else 0} lines")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
