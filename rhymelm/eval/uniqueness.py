"""Uniqueness scoreboard: compare lyric generators on originality and craft.

Every system answers the same public prompt list (eval/prompts.txt). Each
sample file is JSONL: {"system", "prompt", "text"}. Metrics, per system:

  cliche_4gram      share of the sample's 4-grams found in a public rap corpus (lower = fresher)
  cross_model_sim   mean max 3-gram Jaccard vs other systems' answers to the same prompt
                    (lower = says something the other models don't)
  distinct_2/3      unique n-grams / total n-grams across the system's samples
  end_rhyme         share of lines whose end word rhymes with one of the next 3 lines
  multi_rhyme       same, but matching the last TWO vowel sounds (multisyllabic)
  copy_span_max     longest word span copied verbatim from the private training corpus
  copy_8gram        share of 8-grams found in the private corpus (memorization)

Copy metrics need --private-corpus; samples with copy_span >= --leak-span are
listed so they are never committed.

Usage:
    python -m rhymelm.eval.uniqueness reports/samples/*.jsonl \
        --public-corpus lyrics_raw.csv --private-corpus data/private/corpus/train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def ngrams(ws: list[str], n: int) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def load_public_4grams(csv_path: str) -> set[tuple]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    col = "artist_verses" if "artist_verses" in df.columns else df.columns[-1]
    grams: set[tuple] = set()
    for t in df[col].dropna().astype(str):
        grams.update(ngrams(words(t), 4))
    return grams


def load_private_words(path: str) -> list[list[str]]:
    return [words(json.loads(l)["text"]) for l in open(path, encoding="utf-8")]


def longest_copy_span(sample: list[str], private_8: set[tuple], private_docs: list[list[str]]) -> int:
    """Longest verbatim span (in words) shared with any private doc. Seeds on 8-grams."""
    if not private_8 or len(sample) < 8:
        return 0
    best = 0
    joined = [" " + " ".join(d) + " " for d in private_docs]
    i = 0
    while i <= len(sample) - 8:
        if tuple(sample[i:i + 8]) in private_8:
            j = i + 8
            while j < len(sample) and any(" " + " ".join(sample[i:j + 1]) + " " in d for d in joined):
                j += 1
            best = max(best, j - i)
            i = j
        else:
            i += 1
    return best


def rhyme_keys(line: str, cmu: dict) -> tuple[str | None, str | None]:
    """(last stressed-vowel rhyme, last-two-vowels rhyme) for the line's final word."""
    ws = words(line)
    if not ws or ws[-1] not in cmu:
        return None, None
    ph = cmu[ws[-1]][0]
    vowels = [i for i, p in enumerate(ph) if p[-1].isdigit()]
    if not vowels:
        return None, None
    stressed = [i for i in vowels if ph[i][-1] in "12"] or vowels
    single = " ".join(re.sub(r"\d", "", p) for p in ph[stressed[-1]:])
    multi = None
    # two-vowel tail, possibly spanning the previous word
    tail_vowels = [re.sub(r"\d", "", ph[i]) for i in vowels[-2:]]
    if len(tail_vowels) < 2 and len(ws) >= 2 and ws[-2] in cmu:
        prev = [re.sub(r"\d", "", p) for p in cmu[ws[-2]][0] if p[-1].isdigit()]
        tail_vowels = prev[-1:] + tail_vowels
    if len(tail_vowels) == 2:
        multi = " ".join(tail_vowels)
    return single, multi


def rhyme_rates(text: str, cmu: dict) -> tuple[float, float]:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0, 0.0
    keys = [rhyme_keys(l, cmu) for l in lines]
    end = multi = 0
    for i, (s, m) in enumerate(keys):
        nxt = keys[i + 1:i + 4]
        if s and any(s == k[0] for k in nxt):
            end += 1
        if m and any(m == k[1] for k in nxt):
            multi += 1
    return end / len(lines), multi / len(lines)


def family(system: str) -> str:
    """'rhymelm-v3-dpo' -> 'rhymelm'; every other system is its own family."""
    return "rhymelm" if system.startswith("rhymelm") else system


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def score(sample_files: list[str], public_csv: str | None, private_path: str | None,
          leak_span: int = 8) -> dict:
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()

    samples = [json.loads(l) for f in sample_files for l in open(f, encoding="utf-8") if l.strip()]
    public = load_public_4grams(public_csv) if public_csv else set()
    priv_docs = load_private_words(private_path) if private_path else []
    priv_8 = {g for d in priv_docs for g in ngrams(d, 8)}

    by_prompt: dict[str, dict[str, set]] = defaultdict(dict)
    for s in samples:
        by_prompt[s["prompt"]][s["system"]] = set(ngrams(words(s["text"]), 3))

    per_sys: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    leaks = []
    for s in samples:
        ws = words(s["text"])
        g4 = ngrams(ws, 4)
        m = per_sys[s["system"]]
        m["cliche_4gram"].append(sum(g in public for g in g4) / len(g4) if g4 and public else 0.0)
        # compare against OTHER model families only: rhymelm-v2 vs rhymelm-v3 being
        # alike is expected and says nothing about originality vs other models
        others = [v for k, v in by_prompt[s["prompt"]].items() if family(k) != family(s["system"])]
        mine = by_prompt[s["prompt"]][s["system"]]
        m["cross_model_sim"].append(max((jaccard(mine, o) for o in others), default=0.0))
        e, mu = rhyme_rates(s["text"], cmu)
        m["end_rhyme"].append(e)
        m["multi_rhyme"].append(mu)
        m["_words"].append(ws)
        m["lines"].append(sum(1 for l in s["text"].splitlines() if l.strip()))
        if priv_docs:
            g8 = ngrams(ws, 8)
            m["copy_8gram"].append(sum(g in priv_8 for g in g8) / len(g8) if g8 else 0.0)
            span = longest_copy_span(ws, priv_8, priv_docs)
            m["copy_span_max"].append(span)
            if span >= leak_span:
                leaks.append({"system": s["system"], "prompt": s["prompt"], "copy_span": span})

    table = {}
    for sys_name, m in per_sys.items():
        allw = m.pop("_words")
        row = {k: round(sum(v) / len(v), 4) for k, v in m.items() if v and k != "copy_span_max"}
        if m.get("copy_span_max"):
            row["copy_span_max"] = max(m["copy_span_max"])
        for n in (2, 3):
            grams = [g for ws in allw for g in ngrams(ws, n)]
            row[f"distinct_{n}"] = round(len(set(grams)) / len(grams), 4) if grams else 0.0
        row["samples"] = len(allw)
        table[sys_name] = row
    return {"systems": table, "leaks": leaks, "prompts": len(by_prompt)}


def reference_row(path: str, public_csv: str | None) -> dict:
    """Aggregate metrics for a corpus (e.g. Rob's held-out songs). No text is stored."""
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()
    public = load_public_4grams(public_csv) if public_csv else set()
    docs = [json.loads(l) for l in open(path, encoding="utf-8")]
    docs = [d for d in docs if d.get("kind", "lyrics") == "lyrics"]
    m: dict[str, list[float]] = defaultdict(list)
    allw = []
    for d in docs:
        ws = words(d["text"])
        g4 = ngrams(ws, 4)
        m["cliche_4gram"].append(sum(g in public for g in g4) / len(g4) if g4 and public else 0.0)
        e, mu = rhyme_rates(d["text"], cmu)
        m["end_rhyme"].append(e)
        m["multi_rhyme"].append(mu)
        allw.append(ws)
    row = {k: round(sum(v) / len(v), 4) for k, v in m.items() if v}
    grams = [g for ws in allw for g in ngrams(ws, 3)]
    row["distinct_3"] = round(len(set(grams)) / len(grams), 4) if grams else 0.0
    lines = [len(l.split()) for d in docs for l in d["text"].splitlines() if l.strip()]
    row["words_per_line_median"] = sorted(lines)[len(lines) // 2] if lines else 0
    row["docs"] = len(docs)
    row["name"] = Path(path).stem
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--public-corpus", default="lyrics_raw.csv")
    ap.add_argument("--private-corpus", default=None)
    ap.add_argument("--leak-span", type=int, default=8)
    ap.add_argument("--out", default="reports/scoreboard.json")
    ap.add_argument("--reference", default=None,
                    help="private corpus jsonl scored as an aggregate target row (numbers only)")
    ap.add_argument("--reference-name", default="rob-heldout")
    args = ap.parse_args()

    files = [f for pat in args.samples for f in glob.glob(pat)]
    res = score(files, args.public_corpus, args.private_corpus, args.leak_span)
    if args.reference:
        res["reference"] = reference_row(args.reference, args.public_corpus)
        res["reference"]["name"] = args.reference_name
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    cols = ["cliche_4gram", "cross_model_sim", "distinct_3", "end_rhyme", "multi_rhyme", "copy_span_max"]
    print(f"{'system':28}" + "".join(f"{c:>16}" for c in cols))
    for name, row in sorted(res["systems"].items(), key=lambda kv: kv[1].get("cliche_4gram", 1)):
        print(f"{name:28}" + "".join(f"{row.get(c, '-'):>16}" for c in cols))
    if res.get("reference"):
        ref = res["reference"]
        print(f"{'(target) ' + ref['name']:28}" + "".join(f"{ref.get(c, '-'):>16}" for c in cols))
    if res["leaks"]:
        print(f"\nLEAKS (copy span >= {args.leak_span} words; never commit these): {len(res['leaks'])}")


if __name__ == "__main__":
    main()
