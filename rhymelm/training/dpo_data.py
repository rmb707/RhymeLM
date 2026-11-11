"""Build DPO preference pairs from the SFT model's own candidates.

For every title in configs/dpo_titles.txt: sample K verses from base+adapter
(newline-safe sampler), score each with an automatic reward, and pair the best
against the worst. Writes PRIVATE files (candidates can echo Rob's lines):
    data/private/dpo/<name>/candidates.jsonl
    data/private/dpo/<name>/pairs.jsonl      {"prompt", "chosen", "rejected", ...}
and a counts-only summary to reports/dpo/<name>.json.

Reward (what the SFT model lacks, measured against Rob's own lyrics):
  + 1.0 x multisyllabic rhyme rate     (Rob 0.28, SFT model 0.03)
  + 0.7 x end rhyme rate               (Rob 0.20, SFT model 0.03)
  - 3.0 x public-corpus cliche rate
  - 0.5 if fewer than 80% distinct lines (looping)
  disqualified (-1) if: <12 lines, median line length outside 3-10 words,
  or any 6+ word span copied from Rob's training corpus.

Usage:
    python -m rhymelm.training.dpo_data --adapter checkpoints/<run>/adapter --name v2r1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from rhymelm.eval.uniqueness import (longest_copy_span, load_public_4grams, ngrams,
                                     rhyme_rates, words)
from rhymelm.training.sft_qlora import generation_prompt


def read_titles(path: str) -> list[str]:
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def check_disjoint(titles: list[str], eval_prompts: str, private_dir: Path) -> None:
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    ev = {norm(t) for t in read_titles(eval_prompts)}
    rob = {norm(json.loads(l)["title"]) for l in open(private_dir / "notes.jsonl", encoding="utf-8")}
    bad = [t for t in titles if norm(t) in ev or norm(t) in rob]
    if bad:
        raise SystemExit(f"titles overlap eval prompts or Rob's songs: {bad}")


def reward(text: str, cmu, public4: set, priv8: set, priv_docs: list) -> tuple[float, dict]:
    lines = [l for l in text.splitlines() if l.strip()]
    ws = words(text)
    info = {"lines": len(lines)}
    if len(lines) < 12:
        return -1.0, {**info, "why": "short"}
    med = statistics.median(len(l.split()) for l in lines)
    info["words_per_line"] = med
    if not 3 <= med <= 10:
        return -1.0, {**info, "why": "line_shape"}
    span = longest_copy_span(ws, priv8, priv_docs)
    info["copy_span"] = span
    if span >= 6:
        return -1.0, {**info, "why": "copy"}
    end, multi = rhyme_rates(text, cmu)
    g4 = ngrams(ws, 4)
    cliche = sum(g in public4 for g in g4) / len(g4) if g4 else 0.0
    distinct = len({l.strip().lower() for l in lines}) / len(lines)
    r = 1.0 * multi + 0.7 * end - 3.0 * cliche - (0.5 if distinct < 0.8 else 0.0)
    info.update(end_rhyme=round(end, 3), multi_rhyme=round(multi, 3),
                cliche=round(cliche, 4), distinct_lines=round(distinct, 2))
    return round(r, 4), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--titles", default="configs/dpo_titles.txt")
    ap.add_argument("--eval-prompts", default="eval/prompts.txt")
    ap.add_argument("--private", default="data/private")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--min-gap", type=float, default=0.08)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--revision", default="49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--score-only", action="store_true", help="re-score existing candidates")
    args = ap.parse_args()

    private = Path(args.private)
    out_dir = private / "dpo" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    titles = read_titles(args.titles)
    check_disjoint(titles, args.eval_prompts, private)

    cand_path = out_dir / "candidates.jsonl"
    if not args.score_only:
        import torch
        from peft import PeftModel
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  LogitsProcessorList, set_seed)
        from rhymelm.eval.sample_local import clean_verse, newline_safe_penalty

        set_seed(args.seed)
        tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, revision=args.revision,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16, device_map={"": 0})
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        rep = newline_safe_penalty(tok, args.repetition_penalty)

        done = set()
        if cand_path.exists():
            done = {json.loads(l)["prompt"] for l in open(cand_path, encoding="utf-8")}
        with cand_path.open("a", encoding="utf-8") as f:
            for i, title in enumerate(titles, 1):
                if title in done:
                    continue
                ids = tok(generation_prompt(title), return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **ids, do_sample=True, num_return_sequences=args.k,
                        max_new_tokens=args.bars * 16, min_new_tokens=args.bars * 5,
                        temperature=args.temperature, top_p=args.top_p,
                        logits_processor=LogitsProcessorList([rep]),
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                n = ids["input_ids"].shape[1]
                verses = [clean_verse(tok.decode(g[n:], skip_special_tokens=True), args.bars) for g in gen]
                f.write(json.dumps({"prompt": title, "candidates": verses}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}/{len(titles)}] {title}: {len(verses)} candidates")

    # ---- score + pair ----
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()
    public4 = load_public_4grams("lyrics_raw.csv")
    priv_docs = [words(json.loads(l)["text"]) for l in open(private / "corpus" / "train.jsonl", encoding="utf-8")]
    priv8 = {g for d in priv_docs for g in ngrams(d, 8)}

    pairs, all_r, why = [], [], {}
    for line in open(cand_path, encoding="utf-8"):
        row = json.loads(line)
        scored = []
        for v in row["candidates"]:
            r, info = reward(v, cmu, public4, priv8, priv_docs)
            scored.append((r, v, info))
            all_r.append(r)
            if r == -1.0:
                why[info["why"]] = why.get(info["why"], 0) + 1
        scored.sort(key=lambda x: -x[0])
        prompt = generation_prompt(row["prompt"])
        # up to two pairs per title: best vs worst, second vs second-worst
        for (rc, vc, ic), (rr, vr, ir) in [(scored[0], scored[-1]), (scored[1], scored[-2])]:
            if rc <= 0 or rc - rr < args.min_gap or vc == vr:
                continue
            pairs.append({"prompt": prompt, "chosen": vc + "\n", "rejected": vr + "\n",
                          "title": row["prompt"], "r_chosen": rc, "r_rejected": rr})

    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    valid = [r for r in all_r if r > -1.0]
    summary = {
        "name": args.name, "adapter": args.adapter, "titles": len(titles), "k": args.k,
        "candidates": len(all_r), "valid": len(valid), "disqualified": why,
        "reward_mean_valid": round(statistics.mean(valid), 4) if valid else None,
        "pairs": len(pairs),
        "chosen_mean": round(statistics.mean(p["r_chosen"] for p in pairs), 4) if pairs else None,
        "rejected_mean": round(statistics.mean(p["r_rejected"] for p in pairs), 4) if pairs else None,
    }
    Path("reports/dpo").mkdir(parents=True, exist_ok=True)
    Path(f"reports/dpo/{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
