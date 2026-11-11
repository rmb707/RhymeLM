# RhymeLM progress log

Goal: a model trained on Rob's own decade of lyrics that writes the most original
rap of any model. Rob's lyrics never enter this repo. Only code, metrics and model
outputs that pass `rhymelm/eval/publish_samples.py` (no 8+ word copied spans, no
private names) are committed.

Scoreboard: `reports/scoreboard.json`, built by `rhymelm/eval/uniqueness.py` on the
30 fixed titles in `eval/prompts.txt`.

---

## 2026-11-11 (PST): v1, first fine-tune

**Data (private, counts in `reports/corpus_stats.json`).** 971 Obsidian notes were
labeled by content with Codex Sol, because the Apple folder tags were unreliable.
Notes and transcribed voice memos gave 502 documents and 114k words, split
444/34/24 for train/val/test by document.

**Model.** Qwen3-8B-Base (Apache-2.0) with QLoRA r=16, 3 epochs, 9.5 min on an RTX 3080 Ti.
Validation loss fell from 3.66 to 3.56 to 3.55 and was still falling, so the model is undertrained.

**Scoreboard v1:**

| system | cliché 4-gram | distinct-3 | end rhyme | multi rhyme | copied span |
|---|---|---|---|---|---|
| grok-4.7 | 0.0014 | 0.989 | 0.410 | 0.129 | 0 |
| codex-gpt-6.1-sol | 0.0038 | 0.990 | 0.138 | 0.165 | 0 |
| claude-opus-5.5 | 0.0058 | 0.990 | 0.304 | 0.190 | 0 |
| qwen3-8b-base (untuned) | 0.0092 | 0.997 | 0.034 | 0.034 | 0 |
| **rhymelm-v1** | 0.0137 | 0.993 | 0.076 | 0.046 | 0 |
| *target: Rob's held-out songs* | 0.0176 | 0.948 | 0.202 | **0.281** | n/a |

**What it shows:**
- **Rob's multisyllabic rhyme rate (0.28) beats every frontier model** (Claude 0.19,
  Codex 0.16, Grok 0.13). That is the signature to capture. v1 has not learned it.
- v1 writes 12-word lines, but Rob writes 6-word half-bars. Its rhymes land mid-line,
  where the end-rhyme metric does not see them, so line shape is not learned yet.
- The cliché rate against a public rap corpus measures how much a text sounds like
  rap, not quality alone. The LLMs score low because they write literary verse, not
  rap. A blind multi-model judge panel is next, to score originality and craft together.
- 7 of 30 v1 verses stopped at 2 or fewer lines. 136 short fragment documents teach
  early EOS.
- Memorization is clean: 0 copied spans and no private names.

Kimi K3.1 baseline is missing: the hub route hit a rate limit and the direct
instance returned empty runs. It will be added when capacity returns.

**Next (v2):** more epochs and a higher learning rate. Train fragments as their own
`Type: bars` documents but enforce a minimum verse length at generation.
Then the judge panel, followed by preference tuning toward originality.
