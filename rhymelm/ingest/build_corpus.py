"""Assemble the private training corpus from labeled notes and transcripts.

Inputs (all under the private dir):
    notes.jsonl        from rhymelm.ingest.obsidian
    labels.jsonl       one label per note id (rob_lyrics, rob_fragment, ...)
    transcripts.jsonl  from rhymelm.ingest.transcribe (optional)

Outputs:
    corpus/{train,val,test}.jsonl   {"id", "source", "kind", "text"}  -- PRIVATE
    corpus_stats.json               counts only, no text -- safe to commit

Splits are by document and stable (hash of id), so the test set is a fixed
set of Rob's songs the model never sees. Novelty eval uses it.

Usage:
    python -m rhymelm.ingest.build_corpus [--include-prose]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

LYRIC_LABELS = {"rob_lyrics", "rob_fragment"}

# Recordings featuring other artists: their verses would be learned as Rob's
COLLAB = re.compile(r"\b(ft|feat)\b\.?|\bw(/|ith)?\s+rob\b|\bfeaturing\b", re.I)

# Redaction: these never belong in lyrics the model learns to reproduce
PII_PATTERNS = [
    (re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), "[phone]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"\b\d{2,5}\s+\w+(?:\s\w+)?\s(?:st|street|ave|avenue|rd|road|blvd|dr|drive|ln|lane|ct|court|way)\b\.?", re.I), "[address]"),
    (re.compile(r"https?://\S+|www\.\S+"), ""),
    # account / order / tracking numbers: 6+ digit runs, or mixed IDs like AB12C0000001
    (re.compile(r"\b\d{6,}\b"), "[number]"),
    (re.compile(r"\b(?=[A-Za-z0-9]*\d[A-Za-z0-9]*\d[A-Za-z0-9]*\d[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,}\b"), "[id]"),
]


def redact(text: str) -> tuple[str, int]:
    n = 0
    for pat, rep in PII_PATTERNS:
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def _shingles(text: str, k: int = 3) -> set[str]:
    toks = re.findall(r"[a-z']+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def near_dedup(docs: list[dict], threshold: float = 0.6) -> tuple[list[dict], int]:
    """Drop earlier drafts: if two docs overlap >= threshold (containment of the
    shorter in the longer), keep the longer one."""
    docs = sorted(docs, key=lambda d: -len(d["text"]))
    kept: list[tuple[dict, set[str]]] = []
    dropped = 0
    for d in docs:
        sh = _shingles(d["text"])
        if not sh:
            continue
        dup = False
        for _, ksh in kept:
            if len(sh & ksh) / len(sh) >= threshold:
                dup = True
                break
        if dup:
            dropped += 1
        else:
            kept.append((d, sh))
    return [d for d, _ in kept], dropped


def split_of(doc_id: str, val: float = 0.05, test: float = 0.05) -> str:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def build(private: Path, include_prose: bool = False, min_words: int = 12) -> dict:
    notes = {json.loads(l)["id"]: json.loads(l) for l in open(private / "notes.jsonl", encoding="utf-8")}
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(private / "labels.jsonl", encoding="utf-8")}
    want = set(LYRIC_LABELS) | ({"poetry_prose"} if include_prose else set())

    stats: dict = {"notes_total": len(notes), "labels": {}, "redactions": 0}
    for lab in labels.values():
        stats["labels"][lab["label"]] = stats["labels"].get(lab["label"], 0) + 1

    docs: list[dict] = []
    for nid, lab in labels.items():
        if lab["label"] not in want or nid not in notes:
            continue
        text, n = redact(notes[nid]["text"])
        stats["redactions"] += n
        if len(text.split()) < min_words:
            continue
        kind = "fragment" if lab["label"] == "rob_fragment" else (
            "prose" if lab["label"] == "poetry_prose" else "lyrics")
        docs.append({"id": nid, "source": "note", "kind": kind,
                     "title": notes[nid]["title"], "text": text})

    tpath = private / "transcripts.jsonl"
    stats["transcripts_usable"] = 0
    if tpath.exists():
        for l in open(tpath, encoding="utf-8"):
            t = json.loads(l)
            if not t.get("usable"):
                continue
            if COLLAB.search(t["file"]):
                # other artists' verses are on these; no speaker separation yet
                stats["transcripts_collab_excluded"] = stats.get("transcripts_collab_excluded", 0) + 1
                continue
            stats["transcripts_usable"] += 1
            text, n = redact(t["text"])
            stats["redactions"] += n
            title = re.sub(r"\s*\((?:rough )?(?:master|no mix|first verse done)\)|\s+v\d+$", "",
                           Path(t["file"]).stem, flags=re.I)
            docs.append({"id": "audio-" + t["sha256"][:12], "source": "voice_memo",
                         "kind": "lyrics", "title": title, "text": text})

    stats["docs_before_dedup"] = len(docs)
    docs, dropped = near_dedup(docs)
    stats["near_duplicates_dropped"] = dropped

    out = private / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    split_stats = {s: {"docs": 0, "words": 0, "lines": 0} for s in files}
    for d in sorted(docs, key=lambda d: d["id"]):
        s = split_of(d["id"])
        files[s].write(json.dumps(d, ensure_ascii=False) + "\n")
        split_stats[s]["docs"] += 1
        split_stats[s]["words"] += len(d["text"].split())
        split_stats[s]["lines"] += sum(1 for x in d["text"].splitlines() if x.strip())
    for f in files.values():
        f.close()

    stats["splits"] = split_stats
    stats["by_source"] = {}
    stats["by_kind"] = {}
    for d in docs:
        stats["by_source"][d["source"]] = stats["by_source"].get(d["source"], 0) + 1
        stats["by_kind"][d["kind"]] = stats["by_kind"].get(d["kind"], 0) + 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--private", default="data/private")
    ap.add_argument("--include-prose", action="store_true")
    ap.add_argument("--stats-out", default="reports/corpus_stats.json")
    args = ap.parse_args()

    stats = build(Path(args.private), include_prose=args.include_prose)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
