"""Line-by-line rhyme-guided decoding for the fine-tuned lyric model.

Tokens carry spelling, not sound, so a small fine-tune learns Rob's words and
line shape but rarely lands rhymes. This decoder writes one bar at a time:
for every line it samples M candidates from the model (one batched generate),
then picks one:

  * opening line of a rhyme group: the most fluent candidate (mean log-prob)
  * a line that must answer an earlier line (by --scheme): the candidate whose
    end rhymes with it, multisyllabic (last two vowel sounds) preferred over a
    single stressed-vowel rhyme, ties broken by fluency. If none rhymes, the
    most fluent candidate is kept rather than forcing a bad line.

Nothing is constrained at the token level, so the text stays the model's own.
These outputs also serve as "chosen" examples for the next DPO round, which
distills the skill back into the weights.

Usage:
    python -m rhymelm.generation.rhyme_guided --adapter checkpoints/<run>/adapter \
        --system rhymelm-v3-guided --out-dir data/private/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhymelm.eval.uniqueness import rhyme_keys
from rhymelm.training.sft_qlora import generation_prompt

# Which earlier line each line must rhyme with (None = opens a new rhyme group).
SCHEMES = {
    "AABB": lambda i: i - 1 if i % 2 == 1 else None,
    "ABAB": lambda i: i - 2 if i % 4 in (2, 3) else None,
    "AAAA": lambda i: i - 1 if i % 4 != 0 else None,   # held rhyme per 4 bars
}


def pick(cands: list[tuple[str, float]], target: tuple | None, cmu) -> tuple[str, str]:
    """cands: (line, mean_logprob). Returns (line, how) where how in {multi,end,fluent}."""
    best_fluent = max(cands, key=lambda c: c[1])
    if target is None or (target[0] is None and target[1] is None):
        return best_fluent[0], "fluent"
    scored = []
    for line, lp in cands:
        single, multi = rhyme_keys(line, cmu)
        end_word = line.split()[-1].lower().strip(".,!?\"'") if line.split() else ""
        same_word = end_word == target[2]  # rhyming a word with itself is cheating
        tier = 0
        if not same_word:
            if multi and multi == target[1]:
                tier = 2
            elif single and single == target[0]:
                tier = 1
        scored.append((tier, lp, line))
    tier, _, line = max(scored)
    return (line, {2: "multi", 1: "end"}[tier]) if tier else (best_fluent[0], "fluent")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--prompts", default="eval/prompts.txt")
    ap.add_argument("--out-dir", default="data/private/samples")
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--m", type=int, default=8, help="candidates per opening line")
    ap.add_argument("--m-answer", type=int, default=32,
                    help="candidates for lines that must rhyme (rhymes are rare, so cast wider)")
    ap.add_argument("--scheme", default="AABB", choices=sorted(SCHEMES))
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--revision", default="49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import nltk
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, set_seed)

    from rhymelm.eval.sample_local import newline_safe_penalty, read_prompts

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()

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
    newline_ids = {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}

    class EachLineStop(StoppingCriteria):
        """Per-sequence stop once a sequence has emitted a newline after real text."""
        def __init__(self, start: int):
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            gen = input_ids[:, self.start:]
            if gen.shape[1] < 2:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return torch.tensor([int(gen[r, -1]) in newline_ids for r in range(gen.shape[0])],
                                dtype=torch.bool, device=input_ids.device)

    def candidates(context: str, m: int) -> list[tuple[str, float]]:
        ids = tok(context, return_tensors="pt").to(model.device)
        n = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **ids, do_sample=True, num_return_sequences=m, max_new_tokens=32,
                temperature=args.temperature, top_p=args.top_p,
                logits_processor=LogitsProcessorList([rep]),
                stopping_criteria=StoppingCriteriaList([EachLineStop(n)]),
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        logps = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        res = []
        for r in range(out.sequences.shape[0]):
            text = tok.decode(out.sequences[r, n:], skip_special_tokens=True)
            line = text.split("\n")[0].strip()
            if len(line.split()) < 2 or line.lower().startswith(("title:", "type:")):
                continue
            valid = torch.isfinite(logps[r]) & (logps[r] != 0)
            lp = float(logps[r][valid].mean()) if valid.any() else -99.0
            res.append((line, lp))
        return res or [("", -99.0)]

    rule = SCHEMES[args.scheme]
    out_path = Path(args.out_dir) / f"{args.system}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"multi": 0, "end": 0, "fluent": 0}
    with out_path.open("w", encoding="utf-8") as f:
        for title in read_prompts(args.prompts):
            context = generation_prompt(title)
            lines: list[str] = []
            keys: list[tuple] = []
            hows = []
            for i in range(args.bars):
                j = rule(i)
                target = keys[j] if j is not None and j < len(keys) else None
                cands = candidates(context, args.m_answer if target else args.m)
                line, how = pick(cands, target, cmu)
                if not line:
                    break
                lines.append(line)
                s, m = rhyme_keys(line, cmu)
                keys.append((s, m, line.split()[-1].lower().strip(".,!?\"'")))
                hows.append(how)
                context += line + "\n"
            for h in hows:
                totals[h] += 1
            f.write(json.dumps({"system": args.system, "prompt": title, "text": "\n".join(lines),
                                "params": {"decoder": "rhyme_guided", "scheme": args.scheme,
                                           "m": args.m, "m_answer": args.m_answer,
                                           "temperature": args.temperature,
                                           "seed": args.seed},
                                "picks": hows}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{title}: {len(lines)} lines, rhymed {sum(h != 'fluent' for h in hows[1::2])}/{len(hows[1::2])} answer lines")
    print("pick totals:", totals, "->", out_path)


if __name__ == "__main__":
    main()
