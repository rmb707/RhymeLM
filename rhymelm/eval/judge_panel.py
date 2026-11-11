"""Blind multi-model judge panel for lyric samples.

make:      shuffle each prompt's verses under anonymous letters ->
           reports/judging/<round>/packet.jsonl (public: model outputs only)
           data/private/judging/<round>/key.json (letter -> system; judges never see it)
aggregate: read reports/judging/<round>/votes-<judge>.jsonl and un-blind ->
           reports/judging/<round>/results.json (mean rank per system per criterion,
           win rates, and judge agreement)

Vote format, one line per prompt:
  {"prompt": "...", "originality": ["C","A",...], "craft": ["A","C",...]}   best first

Usage:
    python -m rhymelm.eval.judge_panel make --round r1 reports/samples/*.jsonl
    python -m rhymelm.eval.judge_panel aggregate --round r1
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import random
import string
from collections import defaultdict
from pathlib import Path

CRITERIA = ("originality", "craft")


def make(round_name: str, sample_files: list[str], seed: int = 7) -> None:
    by_prompt: dict[str, dict[str, str]] = defaultdict(dict)
    for f in sample_files:
        for line in open(f, encoding="utf-8"):
            s = json.loads(line)
            by_prompt[s["prompt"]][s["system"]] = s["text"]
    rng = random.Random(seed)
    pub = Path("reports/judging") / round_name
    priv = Path("data/private/judging") / round_name
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, str]] = {}
    with open(pub / "packet.jsonl", "w", encoding="utf-8") as out:
        for prompt, systems in by_prompt.items():
            names = sorted(systems)
            rng.shuffle(names)  # fresh order per prompt so letters carry no signal
            letters = string.ascii_uppercase[:len(names)]
            key[prompt] = dict(zip(letters, names))
            out.write(json.dumps({"prompt": prompt,
                                  "verses": {L: systems[n] for L, n in zip(letters, names)}},
                                 ensure_ascii=False) + "\n")
    (priv / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"packet: {len(by_prompt)} prompts x {len(names)} verses -> {pub/'packet.jsonl'}")


def _kendall_agreement(a: list[str], b: list[str]) -> float:
    """Share of item pairs ordered the same way by two rankings (1 = identical)."""
    common = [x for x in a if x in b]
    pairs = list(itertools.combinations(common, 2))
    if not pairs:
        return float("nan")
    same = sum((a.index(x) < a.index(y)) == (b.index(x) < b.index(y)) for x, y in pairs)
    return same / len(pairs)


def aggregate(round_name: str) -> dict:
    pub = Path("reports/judging") / round_name
    key = json.loads((Path("data/private/judging") / round_name / "key.json").read_text(encoding="utf-8"))
    votes: dict[str, dict[str, dict]] = {}
    for f in sorted(pub.glob("votes-*.jsonl")):
        judge = f.stem.removeprefix("votes-")
        votes[judge] = {}
        for line in open(f, encoding="utf-8"):
            v = json.loads(line)
            votes[judge][v["prompt"]] = v

    ranks: dict[str, dict[str, list[float]]] = {c: defaultdict(list) for c in CRITERIA}
    firsts: dict[str, dict[str, int]] = {c: defaultdict(int) for c in CRITERIA}
    per_judge: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: {c: defaultdict(list) for c in CRITERIA})
    invalid = 0
    for judge, by_prompt in votes.items():
        for prompt, v in by_prompt.items():
            mapping = key.get(prompt)
            if not mapping:
                invalid += 1
                continue
            for c in CRITERIA:
                order = v.get(c, [])
                if sorted(order) != sorted(mapping):
                    invalid += 1
                    continue
                n = len(order)
                for pos, letter in enumerate(order):
                    sysname = mapping[letter]
                    # normalized rank: 0 = best, 1 = worst
                    r = pos / (n - 1) if n > 1 else 0.0
                    ranks[c][sysname].append(r)
                    per_judge[judge][c][sysname].append(r)
                firsts[c][mapping[order[0]]] += 1

    agreement = {}
    judges = list(votes)
    for c in CRITERIA:
        vals = []
        for j1, j2 in itertools.combinations(judges, 2):
            for prompt in set(votes[j1]) & set(votes[j2]):
                a, b = votes[j1][prompt].get(c, []), votes[j2][prompt].get(c, [])
                if a and b:
                    vals.append(_kendall_agreement(a, b))
        agreement[c] = round(sum(vals) / len(vals), 3) if vals else None

    res = {
        "round": round_name,
        "judges": judges,
        "invalid_votes": invalid,
        "mean_rank": {c: {s: round(sum(v) / len(v), 3) for s, v in sorted(ranks[c].items(), key=lambda kv: sum(kv[1]) / len(kv[1]))} for c in CRITERIA},
        "first_place": {c: dict(sorted(firsts[c].items(), key=lambda kv: -kv[1])) for c in CRITERIA},
        "by_judge": {j: {c: {s: round(sum(v) / len(v), 3) for s, v in d[c].items()} for c in CRITERIA} for j, d in per_judge.items()},
        "pairwise_agreement": agreement,
        "note": "mean_rank: 0 = always best, 1 = always worst. Judges are blind to system names.",
    }
    (pub / "results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--round", required=True)
    m.add_argument("samples", nargs="+")
    a = sub.add_parser("aggregate")
    a.add_argument("--round", required=True)
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.round, [f for p in args.samples for f in glob.glob(p)])
    else:
        res = aggregate(args.round)
        for c in CRITERIA:
            print(f"\n{c} (mean rank, 0=best):")
            for s, r in res["mean_rank"][c].items():
                print(f"  {s:24} {r:.3f}  firsts={res['first_place'][c].get(s, 0)}")
        print(f"\nagreement {res['pairwise_agreement']}  invalid={res['invalid_votes']}")


if __name__ == "__main__":
    main()
