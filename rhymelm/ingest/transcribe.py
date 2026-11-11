"""Transcribe voice memos linked from vault notes with faster-whisper.

Reads ``notes.jsonl`` (from ``rhymelm.ingest.obsidian``), transcribes each unique
audio file once, and writes ``transcripts.jsonl`` under the same private dir.
Instrumentals and music-only clips are flagged via VAD speech coverage and
words-per-minute so they never enter the corpus.

Usage:
    python -m rhymelm.ingest.transcribe --model large-v3
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Common whisper hallucinations over music / silence
HALLUCINATIONS = re.compile(
    r"(thank(s)? (you )?for watching|subscribe|subtitles by|amara\.org|\bmusic\b\s*$|^\W*$)",
    re.I,
)


def _cuda_dlls() -> None:
    """Expose torch's bundled cuBLAS/cuDNN to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401
        lib = Path(torch.__file__).parent / "lib"
        if lib.exists():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


def load_model(name: str, device: str):
    _cuda_dlls()
    from faster_whisper import WhisperModel
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(name, device=device, compute_type=compute), device


def decode_audio(path: str, sr: int = 16000):
    """Decode any audio to mono float32 at ``sr`` via the ffmpeg CLI.

    Avoids faster-whisper's PyAV path, whose API drifts between av releases.
    """
    import subprocess
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "s16le",
         "-ac", "1", "-ar", str(sr), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def file_sha(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transcribe_file(model, path: str) -> dict:
    segments, info = model.transcribe(
        decode_audio(path),
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,  # limits runaway repetition loops
        no_speech_threshold=0.6,
        word_timestamps=False,
    )
    lines, speech = [], 0.0
    for s in segments:
        t = s.text.strip()
        if not t or HALLUCINATIONS.search(t) or s.no_speech_prob > 0.8:
            continue
        lines.append(t)
        speech += s.end - s.start
    words = sum(len(l.split()) for l in lines)
    dur = float(info.duration or 0.0)
    wpm = words / (dur / 60) if dur else 0.0
    # Repetition check: share of distinct lines
    distinct = len(set(l.lower() for l in lines)) / len(lines) if lines else 0.0
    usable = words >= 25 and wpm >= 20 and distinct >= 0.5
    return {
        "duration_s": round(dur, 1),
        "speech_s": round(speech, 1),
        "words": words,
        "wpm": round(wpm, 1),
        "distinct_line_ratio": round(distinct, 2),
        "usable": usable,
        "text": "\n".join(lines),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notes", default="data/private/notes.jsonl")
    ap.add_argument("--out", default="data/private/transcripts.jsonl")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    audio_to_notes: dict[str, list[str]] = {}
    for line in open(args.notes, encoding="utf-8"):
        r = json.loads(line)
        for a in r["audio"]:
            audio_to_notes.setdefault(a, []).append(r["id"])

    # Same memo is often attached to several notes under different IDs:
    # transcribe each distinct file content once.
    by_sha: dict[str, dict] = {}
    for path, ids in sorted(audio_to_notes.items()):
        e = by_sha.setdefault(file_sha(path), {"audio_path": path, "note_ids": [], "copies": 0})
        e["note_ids"] = sorted(set(e["note_ids"]) | set(ids))
        e["copies"] += 1

    out = Path(args.out)
    done = set()
    if out.exists():  # resume; failed records are retried
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                done.add(r["sha256"])

    model, device = load_model(args.model, args.device)
    print(f"model={args.model} device={device} unique_files={len(by_sha)} already_done={len(done)}")
    kept = [json.loads(l) for l in open(out, encoding="utf-8")] if out.exists() else []
    kept = [r for r in kept if not r.get("error")]
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, (sha, e) in enumerate(by_sha.items(), 1):
            if sha in done:
                continue
            path, ids = e["audio_path"], e["note_ids"]
            rec = {"sha256": sha, "audio_path": path, "note_ids": ids,
                   "copies": e["copies"], "file": Path(path).name}
            try:
                rec.update(transcribe_file(model, path))
            except Exception as e:  # keep going; record the failure
                rec.update({"error": f"{type(e).__name__}: {e}", "usable": False})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(by_sha)}] words={rec.get('words', 0)} "
                  f"wpm={rec.get('wpm', 0)} usable={rec.get('usable')}")


if __name__ == "__main__":
    main()
