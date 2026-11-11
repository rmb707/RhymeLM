# Provenance

Models, data and libraries used by the personal-model pipeline (`rhymelm/ingest`,
`rhymelm/training/sft_qlora.py`, `rhymelm/eval`). Licences read from package
metadata / Hugging Face model cards at pin time (2026-11-11).

## Models

| Model | Revision | Licence | Use |
|---|---|---|---|
| [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Apache-2.0 | Base LLM for QLoRA fine-tune |
| [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | `edaa852ec7e145841d8ffdb056a99866b5f0a478` | MIT | Voice-memo transcription |

Considered and deferred: `Qwen/Qwen3.5-9B-Base` (Apache-2.0). Hybrid linear-attention
layers plus a vision tower; its fast kernels are unavailable on Windows, so it is a
stage-2 candidate for a Linux training host.

## Data

| Data | Licence | In repo? |
|---|---|---|
| Rob's own lyrics (Obsidian notes, voice memos) | All rights reserved (Rob) | **Never.** Lives in gitignored `data/private/` |
| CMU Pronouncing Dictionary (via NLTK) | BSD-style | Downloaded at runtime |
| `lyrics_raw.csv` (Kaggle "Rap Lyrics for NLP") | See Kaggle page; lyrics are third-party copyrighted | Legacy; used only as the public cliché reference in eval. Review before any release. |

Only generated samples that pass the scoreboard's leak check (no span of 8+
words copied from the private corpus) are committed under `reports/samples/`.

## Libraries (pinned in `requirements-train.txt`)

| Package | Version | Licence |
|---|---|---|
| torch (cu128) | 2.11.0 | BSD-3-Clause |
| transformers | 5.22.2 | Apache-2.0 |
| peft | 0.21.2 | Apache-2.0 |
| trl | 1.17.1 | Apache-2.0 |
| bitsandbytes | 0.50.2 | MIT |
| accelerate | 1.15.0 | Apache-2.0 |
| datasets | 5.2.0 | Apache-2.0 |
| huggingface-hub | 2.4.0 | Apache-2.0 |
| faster-whisper | 1.2.1 | MIT |
| ctranslate2 | 4.9.0 | MIT |
| nltk | 3.10.4 | Apache-2.0 |
| pandas | 3.1.1 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause (+ bundled permissive) |
| pyyaml | 6.0.3 | MIT |

External tool: `ffmpeg` (CLI, not bundled) decodes audio for transcription.
