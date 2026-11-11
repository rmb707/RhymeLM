"""Gate: copy model samples from the private dir into reports/samples/ only if safe.

A sample set is publishable when NO sample
  1. shares a verbatim span of >= --leak-span words with the private corpus, and
  2. contains a private proper noun: a capitalized word from Rob's corpus that is
     neither common English nor present in the public rap corpus (family names,
     places, people). These would identify Rob's private life even without a copy.

The whole file is refused on any failure, so a partial leak can't slip through.

Usage:
    python -m rhymelm.eval.publish_samples data/private/samples/rhymelm-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from rhymelm.eval.uniqueness import longest_copy_span, ngrams, words

# Capitalized word incl. apostrophes, so "Didn't" stays whole instead of "Didn"
CAP = re.compile(r"\b[A-Z][a-z]{2,}(?:'[a-z]+)?\b")


def _known(word: str, vocab: set[str]) -> bool:
    """Common/public word, allowing slang spellings and inflections."""
    w = word.lower()
    cands = {w, w.split("'")[0], w.rstrip("s"), w[:-2] if w.endswith("es") else w,
             w + "g" if w.endswith("in") else w,             # lookin -> looking
             w.replace("in'", "ing")}
    return any(c in vocab for c in cands)


def private_names(corpus_dir: Path, public_csv: str) -> set[str]:
    import nltk
    import pandas as pd
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words
    common = {w.lower() for w in nltk_words.words()}
    pub = " ".join(pd.read_csv(public_csv)["artist_verses"].dropna().astype(str)).lower()
    pub_words = set(re.findall(r"[a-z']+", pub))
    # A name stays capitalized mid-line ("me and Jay"); slang is only capitalized
    # because it starts a line ("Boutta ...") and shows up lowercase elsewhere.
    caps: Counter[str] = Counter()
    cap_mid: Counter[str] = Counter()
    lower: Counter[str] = Counter()
    for f in corpus_dir.glob("*.jsonl"):
        for doc in open(f, encoding="utf-8"):
            for ln in json.loads(doc)["text"].splitlines():
                toks = re.findall(r"[A-Za-z][A-Za-z']*", ln)
                for i, t in enumerate(toks):
                    if CAP.fullmatch(t):
                        caps[t] += 1
                        if i > 0:
                            cap_mid[t] += 1
                    elif t.islower():
                        lower[t.capitalize()] += 1
    vocab = common | pub_words
    # Conservative: anything Rob never writes in lowercase stays gated, even if it
    # only ever opens a line (a name can do that too). cap_mid is kept for reports.
    return {w for w in caps if not _known(w, vocab) and lower[w] == 0}


def check(samples_path: Path, corpus_dir: Path, public_csv: str, leak_span: int) -> list[str]:
    priv_docs = [words(json.loads(l)["text"])
                 for f in corpus_dir.glob("*.jsonl") for l in open(f, encoding="utf-8")]
    priv_8 = {g for d in priv_docs for g in ngrams(d, 8)}
    names = private_names(corpus_dir, public_csv)
    problems = []
    for line in open(samples_path, encoding="utf-8"):
        s = json.loads(line)
        span = longest_copy_span(words(s["text"]), priv_8, priv_docs)
        if span >= leak_span:
            problems.append(f"{s['prompt']}: copies {span} words verbatim")
        hit = sorted({w for w in CAP.findall(s["text"]) if w in names})
        if hit:
            problems.append(f"{s['prompt']}: {len(hit)} private proper noun(s)")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples")
    ap.add_argument("--corpus-dir", default="data/private/corpus")
    ap.add_argument("--public-corpus", default="lyrics_raw.csv")
    ap.add_argument("--dest", default="reports/samples")
    ap.add_argument("--leak-span", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.samples)
    problems = check(src, Path(args.corpus_dir), args.public_corpus, args.leak_span)
    if problems:
        print(f"REFUSED {src.name}: {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)  # names themselves are never printed
        sys.exit(1)
    dest = Path(args.dest) / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"published {dest} (no copied spans >= {args.leak_span}, no private names)")


if __name__ == "__main__":
    main()
