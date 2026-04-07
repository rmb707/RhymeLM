"""Generate all RhymeLM visualizations from a trained checkpoint."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from collections import Counter

from rhymelm.cli.generate import load_model
from rhymelm.generation.sampler import generate_verse
from rhymelm.generation.rhyme_sampler import generate_rhyming_verse
from rhymelm.generation.artist_style import get_artist_starters, generate_artist_verse
from rhymelm.evaluation.metrics import evaluate_verse, distinct_n, rhyme_density
from rhymelm.evaluation.interpretability import (
    extract_hidden_states, build_word_boundary_labels,
    build_syllable_labels, train_probe, analyze_lstm_neurons,
)
from rhymelm.data.phonemes import get_cmu_dict, count_syllables
from rhymelm.utils import seed_everything

OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT = "checkpoints/rhymelm_lstm_final.pt"
CSV_PATH = "lyrics_raw.csv"

device = torch.device("cpu")
model, tokenizer, ckpt = load_model(CHECKPOINT, device)
starters = get_artist_starters(CSV_PATH)
cmu = get_cmu_dict()

# Style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.family": "monospace",
})

PALETTE = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ff7b72",
           "#79c0ff", "#ffa657", "#56d364", "#bc8cff", "#ffd700"]


# ═══════════════════════════════════════════════════════════════
# 1. CHARACTER EMBEDDING SPACE
# ═══════════════════════════════════════════════════════════════
print("1/6  Embedding space...")
from sklearn.manifold import TSNE

weights = model.embed.weight.detach().cpu().numpy()
chars = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]

# Filter to printable chars for cleaner viz
mask = [c.isprintable() and c.strip() for i, c in enumerate(chars)]
filtered_w = weights[mask]
filtered_c = [c for c, m in zip(chars, mask) if m]

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(filtered_c) - 1))
coords = tsne.fit_transform(filtered_w)

VOWELS = set("aeiouAEIOU")
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
DIGITS = set("0123456789")

cat_colors = {
    "vowel": "#f78166", "consonant": "#58a6ff",
    "digit": "#7ee787", "punctuation": "#d2a8ff",
}

def classify(c):
    if c in VOWELS: return "vowel"
    if c in CONSONANTS: return "consonant"
    if c in DIGITS: return "digit"
    return "punctuation"

fig, ax = plt.subplots(figsize=(12, 9))
for cat, color in cat_colors.items():
    idx = [i for i, c in enumerate(filtered_c) if classify(c) == cat]
    if idx:
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=80, alpha=0.8,
                   edgecolors="white", linewidth=0.3, label=cat, zorder=2)

for i, ch in enumerate(filtered_c):
    if ch.isalpha() or ch.isdigit():
        ax.annotate(ch, (coords[i, 0], coords[i, 1]), fontsize=8,
                    ha="center", va="bottom", color="white", alpha=0.9,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

ax.legend(loc="upper right", fontsize=11, framealpha=0.8,
          facecolor="#161b22", edgecolor="#30363d")
ax.set_title("Character Embedding Space (t-SNE)", fontsize=16, fontweight="bold", pad=15)
ax.grid(True, alpha=0.15)
ax.set_xlabel("Dimension 1", fontsize=11)
ax.set_ylabel("Dimension 2", fontsize=11)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/embedding_space.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# 2. ARTIST VERSE SHOWCASE
# ═══════════════════════════════════════════════════════════════
print("2/6  Artist showcase...")
artists = ["Eminem", "Kendrick Lamar", "2Pac", "Drake", "Nas", "J. Cole"]
artist_verses = {}
artist_metrics = {}

for art in artists:
    seed_everything(42)
    verse = generate_artist_verse(
        model, tokenizer, device, starters, artist=art,
        num_bars=8, temperature=0.75, top_p=0.95,
        repetition_penalty=1.15, rhyme_scheme="AABB",
    )
    artist_verses[art] = verse
    artist_metrics[art] = evaluate_verse(verse)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for idx, (art, verse) in enumerate(artist_verses.items()):
    ax = axes[idx // 3, idx % 3]
    m = artist_metrics[art]

    lines = [l for l in verse.strip().split("\n") if l.strip()][:8]
    text = "\n".join(lines)

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", fontfamily="monospace",
            color="#e6edf3", linespacing=1.6)

    stats = f"rhyme: {m['rhyme_density']:.2f}  distinct-2: {m['distinct_2']:.2f}  unique: {m['unique_line_ratio']:.2f}"
    ax.text(0.05, 0.05, stats, transform=ax.transAxes, fontsize=8,
            color=PALETTE[idx], fontweight="bold")

    ax.set_title(art, fontsize=14, fontweight="bold", color=PALETTE[idx], pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("RhymeLM — Artist-Style Generation (8 bars, AABB scheme)",
             fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/artist_showcase.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# 3. METRICS RADAR CHART PER ARTIST
# ═══════════════════════════════════════════════════════════════
print("3/6  Artist metrics radar...")
metric_keys = ["rhyme_density", "distinct_1", "distinct_2", "unique_line_ratio", "reasonable_line_ratio"]
labels = ["Rhyme\nDensity", "Distinct-1", "Distinct-2", "Unique\nLines", "Reasonable\nLines"]
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_facecolor("#161b22")
fig.patch.set_facecolor("#0d1117")

for idx, art in enumerate(artists):
    m = artist_metrics[art]
    values = [m.get(k, 0) for k in metric_keys]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=art, color=PALETTE[idx], markersize=5)
    ax.fill(angles, values, alpha=0.08, color=PALETTE[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10, color="#c9d1d9")
ax.set_ylim(0, 1.1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#8b949e")
ax.yaxis.grid(True, color="#30363d", alpha=0.5)
ax.xaxis.grid(True, color="#30363d", alpha=0.5)
ax.spines["polar"].set_color("#30363d")

ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.set_title("Generation Quality by Artist", fontsize=16, fontweight="bold", pad=25, color="#c9d1d9")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/artist_radar.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# 4. TEMPERATURE SWEEP — QUALITY vs CREATIVITY
# ═══════════════════════════════════════════════════════════════
print("4/6  Temperature sweep...")
temps = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
temp_metrics = {t: [] for t in temps}

for t in temps:
    for s in range(5):
        seed_everything(s)
        verse = generate_verse(model, tokenizer, device, prompt="I ",
                               num_bars=8, temperature=t, top_p=0.95)
        temp_metrics[t].append(evaluate_verse(verse))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, metric, label in zip(axes,
    ["distinct_2", "rhyme_density", "unique_line_ratio"],
    ["Distinct-2 (Diversity)", "Rhyme Density", "Unique Line Ratio"]):

    means = [np.mean([m[metric] for m in temp_metrics[t]]) for t in temps]
    stds = [np.std([m[metric] for m in temp_metrics[t]]) for t in temps]

    ax.plot(temps, means, "o-", color="#58a6ff", linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(temps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color="#58a6ff")
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

fig.suptitle("Effect of Temperature on Generation Quality (5 samples each)",
             fontsize=16, fontweight="bold", y=1.04)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/temperature_sweep.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# 5. NEURON SELECTIVITY — WHAT LSTM NEURONS DETECT
# ═══════════════════════════════════════════════════════════════
print("5/6  Neuron selectivity...")
seed_everything(42)
sample_verse = generate_verse(model, tokenizer, device, prompt="I ", num_bars=16,
                               temperature=0.8, top_p=0.95)
neuron_results = analyze_lstm_neurons(model, tokenizer, sample_verse, device, top_k=10)

categories = list(neuron_results.keys())
fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5))
if len(categories) == 1:
    axes = [axes]

for ax, cat in zip(axes, categories):
    neurons, scores = zip(*neuron_results[cat])
    colors = ["#7ee787" if s > 0 else "#ff7b72" for s in scores]
    bars = ax.barh(range(len(neurons)), scores, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(neurons)))
    ax.set_yticklabels([f"N{n}" for n in neurons], fontsize=9)
    ax.set_xlabel("Selectivity", fontsize=10)
    ax.set_title(cat.upper(), fontsize=13, fontweight="bold", color=PALETTE[categories.index(cat)])
    ax.axvline(0, color="#8b949e", linewidth=0.5, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.15)

fig.suptitle("LSTM Neuron Selectivity by Character Category",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/neuron_selectivity.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# 6. PROBING — EMERGENT REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════
print("6/6  Linear probing...")
probe_text = sample_verse * 3  # use more text for probing

hidden_states, chars = extract_hidden_states(model, tokenizer, probe_text, device)

# Word boundary probe
wb_labels = build_word_boundary_labels(probe_text)
wb_probe, wb_acc = train_probe(hidden_states, wb_labels, num_classes=2, epochs=30)

# Syllable probe
syl_labels = build_syllable_labels(probe_text, cmu)
syl_probe, syl_acc = train_probe(hidden_states, syl_labels, num_classes=7, epochs=30)

fig, ax = plt.subplots(figsize=(8, 5))

probes = ["Word Boundary", "Syllable Count"]
accs = [wb_acc, syl_acc]
baselines = [
    1 - np.mean(wb_labels),  # majority class baseline
    Counter(syl_labels).most_common(1)[0][1] / len(syl_labels),
]

x = np.arange(len(probes))
width = 0.35

bars1 = ax.bar(x - width/2, accs, width, label="Probe Accuracy",
               color="#58a6ff", alpha=0.9, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, baselines, width, label="Majority Baseline",
               color="#484f58", alpha=0.7, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1%}", ha="center", fontsize=13, fontweight="bold", color="#58a6ff")
for bar, val in zip(bars2, baselines):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1%}", ha="center", fontsize=11, color="#8b949e")

ax.set_xticks(x)
ax.set_xticklabels(probes, fontsize=13)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Linear Probing: Emergent Representations in LSTM Hidden States",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
ax.grid(True, axis="y", alpha=0.15)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/probing_results.png", bbox_inches="tight")
plt.close()


print(f"\nAll visualizations saved to {OUT_DIR}/")
print(f"  embedding_space.png   — character embedding t-SNE")
print(f"  artist_showcase.png   — 6 artists, 8-bar verses")
print(f"  artist_radar.png      — quality metrics per artist")
print(f"  temperature_sweep.png — temp vs diversity/rhyme/uniqueness")
print(f"  neuron_selectivity.png — which LSTM neurons fire for what")
print(f"  probing_results.png   — emergent word boundary & syllable representations")
