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

### v2 training and a sampling bug (2026-11-11, afternoon PST)

v2 (r32, lr 2e-4, 8 epochs) validation loss by epoch: 3.54, 3.46, **3.455**, 3.54,
3.73, 3.94, 4.14. It overfits after epoch 3 on ~140k tokens. The kept adapter is
the epoch-3 checkpoint, which beats v1 (3.55). The first attempt stopped silently
after epoch 2. It was resumed from `checkpoint-32` with `--resume`.

v2 samples exposed a **sampler bug**. HF's repetition penalty also penalizes the
newline token. Each line break made the next less likely, and lines ran on to 30
to 80 words. That, not the model, is why v1, v2 and the base model wrote 12 to 13
word lines against Rob's 6, and why their end-rhyme scores were low. Fixed with
`LyricRepetitionPenalty`, which exempts newline tokens. Base, v1 and v2 are being
re-sampled, and the v1 row above is superseded.

The leak gate falsely refused v2 on slang and contractions ("Lookin", "Didn't"),
which it had read as names. The gate now normalizes those and still holds 355
private-only proper nouns.

### Re-sampled scoreboard and judge round 1 (2026-11-11, ~3:10 PM PST)

With the newline-safe sampler, v1 and v2 write full 16-bar verses at about 5 words
per line (Rob writes 6). Rhyme is still missing. End rhyme is 0.03 and multisyllabic
rhyme is 0.03, against Rob's 0.20 and 0.28.

**Blind judge panel r1.** Claude Opus 5.5, Codex Sol and Grok 4.7 each ranked all six
systems on all 30 titles (`reports/judging/r1/results.json`, mean normalized rank,
0 = best):

| system | originality | craft |
|---|---|---|
| codex-gpt-6.1-sol | **0.03** | 0.16 |
| grok-4.7 | 0.17 | 0.40 |
| claude-opus-5.5 | 0.40 | **0.04** |
| rhymelm-v2 | 0.76 | 0.76 |
| rhymelm-v1 | 0.76 | 0.74 |
| qwen3-8b-base | 0.88 | 0.90 |

The judges agree about 90% of the time on pairwise orderings. Self-preference is
negligible: every judge, including Claude, put Codex first on originality, and
leaving out each judge's own entry changes no ordering. **RhymeLM beats only the
untuned base so far.** It has Rob's voice but not his rhyme craft.

Caveat: all three judges are LLMs and may share a taste for polished, model-like
writing. Rob's own blind ratings are the decisive test.

### Stage 2: DPO v3 and rhyme-guided decoding (2026-11-11, 3:30–5:00 PM PST)

- **v3 = DPO on v2.** It used 147 pairs from v2's own candidates, scored by a rhyme
  reward. On that reward Rob's held-out lyrics score 0.31 and v2 scores -0.03. On
  training pairs v3 prefers the chosen verse 94–100% of the time; on held-out pairs
  it is at chance. End rhyme on the eval titles rose from 0.032 to 0.041; the gain
  is small.
- **Rhyme-guided decoding** (`rhymelm/generation/rhyme_guided.py`) writes one bar at
  a time. It samples line candidates and keeps the one that rhymes with the line it
  answers (AABB), preferring two-vowel rhymes. No token is forced.

| v3 decoding | candidates per rhyming line | rhyming lines landed | end rhyme | multisyllabic rhyme |
|---|---|---|---|---|
| plain sampling | n/a | n/a | 0.041 | 0.030 |
| guided | 8 | 40 / 240 | 0.077 | 0.088 |
| **guided** | **32** | **107 / 240** | **0.158** | **0.188** |
| *Rob's held-out lyrics (target)* | | | *0.202* | *0.281* |

Multisyllabic rhyme now ties Claude Opus 5.5 (0.190) and beats Codex Sol (0.165) and
Grok 4.7 (0.129), while keeping the lowest cross-model similarity. The samples read
coherently, but the rhymes are simple and cluster in the first bars.

**Next:**
1. Use guided-32 verses on the 80 DPO titles as "chosen" and plain samples as
   "rejected" (expert iteration). This trains the rhyming into the weights so plain
   sampling improves too.
2. Judge round 2 with the guided outputs.
3. Judge-in-the-loop preference data. Rob's blind rating is still pending. Train fragments as their own
`Type: bars` documents but enforce a minimum verse length at generation.
Then the judge panel, followed by preference tuning toward originality.
