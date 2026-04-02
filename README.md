RhymeLM
====================================

A character-level language model that generates 16-bar rap verses — built from
scratch in PyTorch as a comprehensive exploration of neural network architecture,
training dynamics, and interpretability.

RhymeLM uses a **dual-corpus training approach**: an English dictionary for
vocabulary grounding (teaching the model what words look like) and rap lyrics
for style, structure, and rhyme schemes (teaching it how artists flow).

```
                        ┌─────────────────┐
    lyrics_raw.csv ───► │  Dual Corpus    │
    English dict   ───► │  Builder        │
    CMU phonemes   ───► │                 │
                        └───────┬─────────┘
                                │ character stream
                        ┌───────▼─────────┐
                        │  Char / BPE     │
                        │  Tokenizer      │
                        └───────┬─────────┘
                                │ token indices
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
   ┌─────▼──────┐   ┌──────────▼─────────┐   ┌───────▼────────┐
   │   LSTM     │   │  LSTM + Attention  │   │  Transformer   │
   │  (2L,512h) │   │  (LSTM + MHA)      │   │  (from scratch)│
   └─────┬──────┘   └──────────┬─────────┘   └───────┬────────┘
         │                      │                      │
         │  ┌───────────────────┤                      │
         │  │ + Phoneme Head    │                      │
         │  │   (multi-task)    │                      │
         └──┴──────────┬───────┘──────────────────────┘
                       │ logits
          ┌────────────▼────────────┐
          │  Sampler                │
          │  temp / top-k / top-p   │
          │  + rhyme constraints    │
          │  + repetition penalty   │
          └────────────┬────────────┘
                       │
                  16-bar verse
```

---

## Quick Start

```bash
pip install -e .

# Train (LSTM baseline)
python train.py --config configs/lstm_base.yaml

# Train other architectures
python train.py --config configs/lstm_attention.yaml
python train.py --config configs/transformer_char.yaml

# Generate
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt \
                   --prompt "Yeah, " --temperature 0.8 --top-p 0.95

# Generate with rhyme scheme
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt \
                   --rhyme-scheme AABB --bars 16

# Generate in artist style
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt \
                   --artist "Eminem" --rhyme-scheme ABAB --eval
```

---

## Model Architectures

### LSTM Baseline
Character embedding (256-dim) → 2-layer LSTM (512 hidden) → linear head.
The recurrent hidden state carries context across the sequence, modeling
the sequential flow of verse.

### LSTM + Phoneme Head (Multi-Task)
Same LSTM backbone with an auxiliary head that predicts CMU phoneme sequences
at word boundaries. The shared representation is forced to encode pronunciation
information, improving rhyme coherence. Demonstrates **multi-task learning**.

### LSTM + Self-Attention Hybrid
Adds multi-head self-attention on top of LSTM output with residual connection
and layer norm. The LSTM handles local sequential modeling; attention enables
looking back at distant positions (like a rhyming word from bars ago).

### Transformer (From Scratch)
Decoder-only transformer with every component hand-built:
- **Scaled dot-product attention**: `softmax(QK^T / sqrt(d_k)) V`
- **Multi-head attention**: split/concat/project with causal mask
- **Positional encoding**: sinusoidal (Vaswani) or learned (GPT-2 style)
- **Pre-norm blocks**: LayerNorm before attention and FFN (GPT-2 convention)
- **GELU activation** in feed-forward layers
- **KV-cache** for O(n)-per-step autoregressive generation
- **Weight tying** between token embedding and output projection

---

## Training Features

- **Gradient accumulation** — effective batch sizes larger than VRAM allows
- **Mixed precision (AMP)** — automatic GPU detection (CUDA / ROCm / MPS)
- **Linear warmup + cosine decay** — LR schedule with configurable warmup
- **Decoupled weight decay** — applied only to weight matrices, not biases or norms
- **Gradient clipping** — max_norm=1.0
- **Label smoothing** — configurable cross-entropy smoothing
- **YAML configs** — full experiment reproducibility

## Generation Features

- Temperature scaling, top-k, nucleus (top-p) sampling
- Repetition penalty
- **Rhyme scheme templates** (AABB, ABAB, ABBA) — constrained decoding using CMU dict
- **Artist-style generation** — primes model with artist-specific opening lines
- KV-cache for fast transformer inference

## Evaluation & Interpretability

- **Perplexity** — standard LM metric
- **Distinct-n** — lexical diversity (unique n-grams / total n-grams)
- **Rhyme density** — fraction of line endings that rhyme (CMU dict)
- **Verse structure score** — line count, line length, repetition detection
- **Linear probing** — classify hidden state features (word boundaries, syllables, phonemes)
- **LSTM neuron analysis** — find units selective for vowels, newlines, etc.
- **Attention head classification** — identify local, structural, and rhyme heads
- **Embedding visualization** — t-SNE/UMAP of character embeddings by category
- **Ablation framework** — systematic comparison with error bars across seeds

---

## Project Structure

```
rhymelm/
    config.py                   # Dataclass-based YAML configuration
    utils.py                    # Device detection, seeding, param helpers
    data/
        corpus.py               # Dual-corpus builder, verse extraction
        tokenizer.py            # Character-level tokenizer
        bpe.py                  # Byte-Pair Encoding tokenizer (from scratch)
        dataset.py              # PyTorch Dataset/DataLoader
        phonemes.py             # CMU dict, rhyme detection, syllable counting, trie
    models/
        base.py                 # Abstract base for all model variants
        lstm.py                 # LSTM language model
        lstm_phoneme.py         # LSTM + phoneme auxiliary head
        lstm_attention.py       # LSTM + self-attention hybrid
        transformer.py          # Decoder-only transformer (hand-built)
    training/
        trainer.py              # Training loop (AMP, grad accum, scheduling)
        phoneme_trainer.py      # Multi-task trainer (char + phoneme loss)
        experiment.py           # JSON-based experiment tracking
    generation/
        sampler.py              # Temperature, top-k, top-p, repetition penalty
        rhyme_sampler.py        # Rhyme-constrained generation with scheme templates
        artist_style.py         # Artist-conditioned generation
    evaluation/
        metrics.py              # Perplexity, distinct-n, rhyme density
        comparison.py           # Multi-model comparison framework
        interpretability.py     # Probing classifiers, neuron/attention analysis
        ablation.py             # Ablation study framework
    visualization/
        training_plots.py       # Training dynamics dashboard
        embedding_viz.py        # Character embedding space (t-SNE/UMAP)
        attention_viz.py        # Attention weight heatmaps
    cli/
        train.py                # CLI training entry point
        generate.py             # CLI generation entry point
configs/
    lstm_base.yaml              # LSTM baseline
    lstm_phoneme.yaml           # LSTM + phoneme multi-task
    lstm_attention.yaml         # LSTM + attention hybrid
    transformer_char.yaml       # Transformer (character-level)
notebooks/
    01_train_lstm.ipynb         # Interactive training walkthrough
    02_evaluate_and_visualize.ipynb  # Analysis and visualization
train.py                        # python train.py [args]
generate.py                     # python generate.py [args]
RhymeLM_v2.ipynb                # Original standalone notebook
```

---

## Neural Network Concepts Demonstrated

| Concept | Implementation |
|---|---|
| LSTM / recurrent networks | `models/lstm.py` |
| Self-attention mechanism | `models/lstm_attention.py` |
| Multi-head attention (hand-built) | `models/transformer.py` |
| Transformer architecture | `models/transformer.py` |
| Positional encoding (sinusoidal + learned) | `models/transformer.py` |
| Causal masking | `models/transformer.py` |
| KV-cache optimization | `models/transformer.py` |
| Multi-task learning | `models/lstm_phoneme.py`, `training/phoneme_trainer.py` |
| Constrained decoding | `generation/rhyme_sampler.py` |
| BPE tokenization | `data/bpe.py` |
| Mixed precision training (AMP) | `training/trainer.py` |
| Gradient accumulation | `training/trainer.py` |
| LR warmup + cosine decay | `training/trainer.py` |
| Label smoothing | `training/trainer.py` |
| Decoupled weight decay | `utils.py` |
| Xavier initialization | `models/base.py` |
| Gradient clipping | `training/trainer.py` |
| Dropout regularization | All models |
| Nucleus / top-k sampling | `generation/sampler.py` |
| Residual connections + LayerNorm | `models/lstm_attention.py`, `models/transformer.py` |
| GELU activation | `models/transformer.py` |
| Weight tying | `models/transformer.py` |
| Probing classifiers | `evaluation/interpretability.py` |
| Attention head analysis | `evaluation/interpretability.py` |
| Neuron selectivity analysis | `evaluation/interpretability.py` |
| Ablation methodology | `evaluation/ablation.py` |
| Experiment tracking | `training/experiment.py` |

---

## Dataset

Training data: a CSV of rap lyrics with columns `artist` and `artist_verses`.

```
lyrics_raw.csv
```

Example dataset: [Rap Lyrics for NLP](https://www.kaggle.com/datasets/ceebloop/rap-lyrics-for-nlp) on Kaggle.

---

## Dependencies

```bash
pip install -e .                  # core (torch, pandas, numpy, nltk, tqdm, matplotlib, pyyaml)
pip install -e ".[viz]"           # + umap-learn, scikit-learn
pip install -e ".[dev]"           # + jupyter
pip install -e ".[viz,dev]"       # everything
```

---

## License

MIT License. Dataset license follows the terms provided on Kaggle.
