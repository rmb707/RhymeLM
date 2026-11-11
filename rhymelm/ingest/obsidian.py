"""Extract candidate lyric notes from an Obsidian vault.

Walks the vault, strips frontmatter and Markdown/Obsidian markup, resolves
embedded voice memos, scores each note for lyric-likeness and writes one JSON
record per note. Output is private training data: write it only under a
gitignored directory (default ``data/private/``). Nothing here prints note text.

Usage:
    python -m rhymelm.ingest.obsidian --vault "/path/to/Obsidian Vault"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote

SKIP_DIRS = {"Agent Memory", "Agent Memory.nosync", ".obsidian", ".git", ".trash", "Note Sync"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".caf"}

FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
MD_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
WIKI_EMBED = re.compile(r"!?\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
HEADING = re.compile(r"^#{1,6}\s+")
URL = re.compile(r"https?://\S+|www\.\S+")

SMART = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "})


@dataclass
class NoteRecord:
    id: str
    rel_path: str
    title: str
    apple_folder: str
    text: str
    audio: list[str] = field(default_factory=list)
    n_lines: int = 0
    n_words: int = 0
    lyric_score: float = 0.0
    end_rhyme_rate: float = 0.0
    text_sha: str = ""


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    raw = raw.lstrip("\ufeff")
    m = FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, raw[m.end():]


def _index_audio(vault: Path) -> dict[str, Path]:
    """Map 'Attachments/...' suffixes to real files (first copy wins)."""
    idx: dict[str, Path] = {}
    for p in vault.rglob("*"):
        if p.suffix.lower() in AUDIO_EXT and p.is_file():
            parts = p.relative_to(vault).parts
            if "Attachments" in parts:
                key = "/".join(parts[parts.index("Attachments"):])
                idx.setdefault(key, p)
    return idx


def clean_body(body: str, title: str, audio_idx: dict[str, Path]) -> tuple[str, list[str]]:
    """Return (clean text, resolved audio paths)."""
    audio: list[str] = []

    def link(m: re.Match) -> str:
        target = unquote(m.group(2)).replace("\\", "/")
        if Path(target.split("?")[0]).suffix.lower() in AUDIO_EXT:
            if "Attachments/" in target:
                hit = audio_idx.get(target[target.index("Attachments/"):])
                if hit:
                    audio.append(str(hit))
            return ""
        if m.group(0).startswith("!"):
            return ""  # images and other embeds carry no lyric text
        return m.group(1)

    text = unicodedata.normalize("NFKC", body).translate(SMART)
    text = MD_LINK.sub(link, text)
    text = WIKI_EMBED.sub(lambda m: "" if m.group(0).startswith("!") else (m.group(2) or m.group(1)), text)

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if HEADING.match(line):
            continue  # Apple import puts the title as an H1
        lines.append(line)
    # Apple import repeats the title as the first body line
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() == title.strip().lower():
        lines.pop(0)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, audio


def score_text(text: str, cmu: dict | None) -> tuple[float, float, int, int]:
    """Lyric-likeness heuristic: short lines, no URLs/numbers/bullets, end rhymes."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    n = len(lines)
    if n == 0:
        return 0.0, 0.0, 0, 0
    short = sum(1 for l in lines if 8 <= len(l) <= 90) / n
    urls = sum(1 for l in lines if URL.search(l)) / n
    digits = sum(1 for l in lines if sum(c.isdigit() for c in l) > 0.3 * len(l)) / n
    bullets = sum(1 for l in lines if re.match(r"^([-*•]|\d+[.)]|\[[ x]\]|\|)", l)) / n
    rhyme = 0.0
    if cmu is not None and n >= 2:
        from rhymelm.data.phonemes import get_rhyme_suffix
        sufs = []
        for l in lines:
            w = re.findall(r"[A-Za-z']+", l)
            sufs.append(get_rhyme_suffix(w[-1], cmu) if w else None)
        hits = 0
        for i, s in enumerate(sufs):
            if s and any(s == t for t in sufs[i + 1:i + 4] if t):
                hits += 1
        rhyme = hits / n
    score = short - 2 * urls - 1.5 * digits - bullets + 0.8 * rhyme - (0.3 if n < 4 else 0)
    words = sum(len(l.split()) for l in lines)
    return round(score, 3), round(rhyme, 3), n, words


def extract(vault: Path, use_cmu: bool = True) -> list[NoteRecord]:
    cmu = None
    if use_cmu:
        from rhymelm.data.phonemes import get_cmu_dict
        cmu = get_cmu_dict()
    audio_idx = _index_audio(vault)
    records: list[NoteRecord] = []
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault)
        if rel.parts[0] in SKIP_DIRS:
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(raw)
        title = p.stem
        text, audio = clean_body(body, title, audio_idx)
        score, rhyme, n, words = score_text(text, cmu)
        rel_s = rel.as_posix()
        records.append(NoteRecord(
            id=hashlib.sha1(rel_s.encode()).hexdigest()[:12],
            rel_path=rel_s,
            title=title,
            apple_folder=meta.get("apple_folder", ""),
            text=text,
            audio=sorted(set(audio)),
            n_lines=n,
            n_words=words,
            lyric_score=score,
            end_rhyme_rate=rhyme,
            text_sha=hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode()).hexdigest(),
        ))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", required=True)
    ap.add_argument("--out", default="data/private/notes.jsonl")
    ap.add_argument("--no-cmu", action="store_true")
    args = ap.parse_args()

    recs = extract(Path(args.vault), use_cmu=not args.no_cmu)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    dupes = len(recs) - len({r.text_sha for r in recs if r.text})
    print(f"notes={len(recs)} words={sum(r.n_words for r in recs):,} "
          f"with_audio={sum(1 for r in recs if r.audio)} exact_dupes={dupes} -> {out}")


if __name__ == "__main__":
    main()
