"""RhymeLM Live Dashboard — interactive verse generation and model analysis."""

import random
import io
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import gradio as gr

from rhymelm.cli.generate import load_model
from rhymelm.generation.sampler import generate_verse
from rhymelm.generation.rhyme_sampler import generate_rhyming_verse
from rhymelm.generation.artist_style import get_artist_starters, generate_artist_verse
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.evaluation.interpretability import (
    extract_hidden_states, build_word_boundary_labels,
    build_syllable_labels, train_probe, analyze_lstm_neurons,
)
from rhymelm.data.phonemes import get_cmu_dict, get_rhyme_suffix
from rhymelm.generation.originality import OriginalityFilter
from rhymelm.data.corpus import load_dictionary, load_lyrics, extract_verses, build_dual_corpus
from rhymelm.utils import seed_everything

# ── Global state ──
CHECKPOINT = "checkpoints/rhymelm_original_final.pt"
CSV_PATH = "lyrics_cleaned.csv"
LYRICS_COL = "lyrics_clean"
DEVICE = torch.device("cpu")

print("Loading model...")
MODEL, TOKENIZER, CKPT = load_model(CHECKPOINT, DEVICE)
STARTERS = get_artist_starters(CSV_PATH, lyrics_column=LYRICS_COL)
CMU = get_cmu_dict()
ARTISTS = sorted(STARTERS.keys())

# Build originality filter from training corpus
print("Building originality filter...")
_lyrics = load_lyrics(CSV_PATH, LYRICS_COL)
_verses = extract_verses(_lyrics)
_corpus = "\n\n".join(_verses)
ORIG_FILTER = OriginalityFilter(_corpus, ngram_size=25)

# Dark theme for matplotlib
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 120,
    "font.family": "monospace",
})
PAL = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ff7b72",
       "#79c0ff", "#ffa657", "#56d364", "#bc8cff", "#ffd700"]


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════
# TAB 1: GENERATE VERSE
# ═══════════════════════════════════════════════════════════════
def generate_tab(prompt, artist, rhyme_scheme, temperature, top_p, rep_penalty, bars, seed_val, use_orig_filter):
    if seed_val >= 0:
        seed_everything(int(seed_val))

    of = ORIG_FILTER if use_orig_filter else None

    if artist and artist != "None":
        verse = generate_artist_verse(
            MODEL, TOKENIZER, DEVICE, STARTERS,
            artist=artist, num_bars=int(bars),
            temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty,
            rhyme_scheme=rhyme_scheme if rhyme_scheme != "free" else None,
            originality_filter=of,
        )
    elif rhyme_scheme and rhyme_scheme != "free":
        verse = generate_rhyming_verse(
            MODEL, TOKENIZER, DEVICE,
            prompt=prompt or " ", num_bars=int(bars),
            temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty,
            rhyme_scheme=rhyme_scheme,
            originality_filter=of,
        )
    else:
        verse = generate_verse(
            MODEL, TOKENIZER, DEVICE,
            prompt=prompt or " ", num_bars=int(bars),
            temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty,
            originality_filter=of,
        )

    metrics = evaluate_verse(verse)

    # Highlight rhyming line endings
    lines = verse.strip().split("\n")
    endings = []
    for line in lines:
        words = line.split()
        if words:
            suffix = get_rhyme_suffix(words[-1], CMU)
            endings.append((words[-1], suffix))
        else:
            endings.append(("", None))

    # Color code rhyming pairs
    suffix_groups = {}
    for i, (word, suffix) in enumerate(endings):
        if suffix:
            suffix_groups.setdefault(suffix, []).append(i)

    rhyme_annotations = []
    color_idx = 0
    for suffix, line_indices in suffix_groups.items():
        if len(line_indices) > 1:
            for li in line_indices:
                rhyme_annotations.append((li, PAL[color_idx % len(PAL)]))
            color_idx += 1

    # Originality check: how many generated lines are copied from training?
    lines = [l.strip() for l in verse.strip().split("\n") if l.strip()]
    copied = sum(1 for l in lines if ORIG_FILTER.is_line_copied_fast(l))
    orig_score = 1 - (copied / max(len(lines), 1))

    # Build metrics display
    metrics_md = f"""| Metric | Value |
|---|---|
| **Originality** | **{orig_score:.0%}** ({copied} of {len(lines)} lines found in training) |
| Rhyme Density | **{metrics['rhyme_density']:.2%}** |
| Distinct-1 | {metrics['distinct_1']:.3f} |
| Distinct-2 | {metrics['distinct_2']:.3f} |
| Distinct-3 | {metrics['distinct_3']:.3f} |
| Unique Lines | {metrics['unique_line_ratio']:.2%} |
| Lines | {metrics['num_lines']} |
| Avg Line Length | {metrics['avg_line_length']} chars |"""

    return verse, metrics_md


# ═══════════════════════════════════════════════════════════════
# TAB 2: ARTIST COMPARISON
# ═══════════════════════════════════════════════════════════════
def artist_comparison_tab(selected_artists, temperature, rhyme_scheme, num_verses):
    if not selected_artists:
        selected_artists = ARTISTS[:4]

    results = {}
    for art in selected_artists:
        art_metrics = []
        for s in range(int(num_verses)):
            seed_everything(s)
            verse = generate_artist_verse(
                MODEL, TOKENIZER, DEVICE, STARTERS,
                artist=art, num_bars=8,
                temperature=temperature, top_p=0.95,
                repetition_penalty=1.15,
                rhyme_scheme=rhyme_scheme if rhyme_scheme != "free" else None,
            )
            art_metrics.append(evaluate_verse(verse))
        results[art] = art_metrics

    # Radar chart
    metric_keys = ["rhyme_density", "distinct_1", "distinct_2", "unique_line_ratio", "reasonable_line_ratio"]
    labels = ["Rhyme", "Dist-1", "Dist-2", "Unique", "Reasonable"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b22")

    for idx, art in enumerate(selected_artists):
        means = [np.mean([m[k] for m in results[art]]) for k in metric_keys]
        means += means[:1]
        ax.plot(angles, means, "o-", linewidth=2.5, label=art, color=PAL[idx % len(PAL)], markersize=6)
        ax.fill(angles, means, alpha=0.1, color=PAL[idx % len(PAL)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color="#c9d1d9")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=9, color="#8b949e")
    ax.yaxis.grid(True, color="#30363d", alpha=0.5)
    ax.xaxis.grid(True, color="#30363d", alpha=0.5)
    ax.spines["polar"].set_color("#30363d")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title(f"Generation Quality ({int(num_verses)} samples/artist)",
                 fontsize=15, fontweight="bold", pad=25, color="#c9d1d9")

    radar_img = fig_to_image(fig)

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(selected_artists))
    width = 0.2
    for mi, (mk, label) in enumerate(zip(["rhyme_density", "distinct_2", "unique_line_ratio"],
                                          ["Rhyme", "Distinct-2", "Unique"])):
        vals = [np.mean([m[mk] for m in results[a]]) for a in selected_artists]
        errs = [np.std([m[mk] for m in results[a]]) for a in selected_artists]
        ax2.bar(x + mi * width, vals, width, yerr=errs, label=label,
                color=PAL[mi], alpha=0.85, edgecolor="white", linewidth=0.3, capsize=3)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(selected_artists, fontsize=11)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
    ax2.set_title("Key Metrics by Artist (mean +/- std)", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.15)
    ax2.set_ylim(0, 1.2)

    bar_img = fig_to_image(fig2)
    return radar_img, bar_img


# ═══════════════════════════════════════════════════════════════
# TAB 3: TEMPERATURE EXPLORER
# ═══════════════════════════════════════════════════════════════
def temperature_explorer_tab(prompt, num_samples):
    temps = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2]
    all_data = {t: [] for t in temps}

    for t in temps:
        for s in range(int(num_samples)):
            seed_everything(s * 100 + int(t * 10))
            verse = generate_verse(MODEL, TOKENIZER, DEVICE, prompt=prompt or "I ",
                                   num_bars=8, temperature=t, top_p=0.95)
            all_data[t].append(evaluate_verse(verse))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, metric, label, color in zip(axes,
        ["distinct_2", "rhyme_density", "unique_line_ratio"],
        ["Lexical Diversity (Distinct-2)", "Rhyme Density", "Unique Line Ratio"],
        [PAL[0], PAL[1], PAL[2]]):

        means = [np.mean([m[metric] for m in all_data[t]]) for t in temps]
        stds = [np.std([m[metric] for m in all_data[t]]) for t in temps]

        ax.plot(temps, means, "o-", color=color, linewidth=2.5, markersize=8, zorder=3)
        ax.fill_between(temps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold", color=color)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.1)

    fig.suptitle(f"Temperature vs Quality ({int(num_samples)} samples each)",
                 fontsize=16, fontweight="bold", y=1.04)
    plt.tight_layout()
    return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════
# TAB 4: MODEL INTERNALS
# ═══════════════════════════════════════════════════════════════
def internals_tab():
    # Embedding space
    from sklearn.manifold import TSNE
    weights = MODEL.embed.weight.detach().cpu().numpy()
    chars = [TOKENIZER.itos[i] for i in range(TOKENIZER.vocab_size)]

    mask = [c.isprintable() and c.strip() for c in chars]
    fw = weights[mask]
    fc = [c for c, m in zip(chars, mask) if m]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fc) - 1))
    coords = tsne.fit_transform(fw)

    VOWELS = set("aeiouAEIOU")
    CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    cat_colors = {"vowel": "#f78166", "consonant": "#58a6ff", "digit": "#7ee787", "punct": "#d2a8ff"}
    def classify(c):
        if c in VOWELS: return "vowel"
        if c in CONSONANTS: return "consonant"
        if c in set("0123456789"): return "digit"
        return "punct"

    fig1, ax = plt.subplots(figsize=(10, 8))
    for cat, color in cat_colors.items():
        idx = [i for i, c in enumerate(fc) if classify(c) == cat]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=70, alpha=0.8,
                       edgecolors="white", linewidth=0.3, label=cat, zorder=2)
    for i, ch in enumerate(fc):
        if ch.isalpha():
            ax.annotate(ch, (coords[i, 0], coords[i, 1]), fontsize=8, ha="center", va="bottom",
                        color="white", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
    ax.set_title("Character Embedding Space (t-SNE)", fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.15)
    embed_img = fig_to_image(fig1)

    # Neuron selectivity
    seed_everything(42)
    text = generate_verse(MODEL, TOKENIZER, DEVICE, prompt="I ", num_bars=16, temperature=0.8, top_p=0.95)
    nr = analyze_lstm_neurons(MODEL, TOKENIZER, text, DEVICE, top_k=8)
    cats = list(nr.keys())

    fig2, axes = plt.subplots(1, len(cats), figsize=(4 * len(cats), 4.5))
    if len(cats) == 1: axes = [axes]
    for ax, cat in zip(axes, cats):
        neurons, scores = zip(*nr[cat])
        colors = ["#7ee787" if s > 0 else "#ff7b72" for s in scores]
        ax.barh(range(len(neurons)), scores, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(neurons)))
        ax.set_yticklabels([f"N{n}" for n in neurons], fontsize=8)
        ax.set_title(cat.upper(), fontsize=12, fontweight="bold", color=PAL[cats.index(cat) % len(PAL)])
        ax.axvline(0, color="#8b949e", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.15)
    fig2.suptitle("LSTM Neuron Selectivity", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    neuron_img = fig_to_image(fig2)

    # Probing
    hidden_states, _ = extract_hidden_states(MODEL, TOKENIZER, text * 2, DEVICE)
    text2 = text * 2
    wb_labels = build_word_boundary_labels(text2)
    syl_labels = build_syllable_labels(text2, CMU)
    _, wb_acc = train_probe(hidden_states, wb_labels, 2, epochs=30)
    _, syl_acc = train_probe(hidden_states, syl_labels, 7, epochs=30)

    from collections import Counter
    fig3, ax = plt.subplots(figsize=(7, 4.5))
    probes = ["Word Boundary", "Syllable Count"]
    accs = [wb_acc, syl_acc]
    baselines = [1 - np.mean(wb_labels), Counter(syl_labels).most_common(1)[0][1] / len(syl_labels)]
    x = np.arange(2)
    ax.bar(x - 0.18, accs, 0.35, label="Probe", color="#58a6ff", alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.bar(x + 0.18, baselines, 0.35, label="Baseline", color="#484f58", alpha=0.7, edgecolor="white", linewidth=0.3)
    for i in range(2):
        ax.text(i - 0.18, accs[i] + 0.03, f"{accs[i]:.0%}", ha="center", fontsize=13, fontweight="bold", color="#58a6ff")
        ax.text(i + 0.18, baselines[i] + 0.03, f"{baselines[i]:.0%}", ha="center", fontsize=11, color="#8b949e")
    ax.set_xticks(x)
    ax.set_xticklabels(probes, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Emergent Representations (Linear Probes)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, axis="y", alpha=0.15)
    plt.tight_layout()
    probe_img = fig_to_image(fig3)

    return embed_img, neuron_img, probe_img


# ═══════════════════════════════════════════════════════════════
# BUILD GRADIO APP
# ═══════════════════════════════════════════════════════════════

dark_css = """
.gradio-container { max-width: 1200px !important; }
.verse-output textarea { font-family: monospace !important; font-size: 14px !important; line-height: 1.7 !important; }
"""

with gr.Blocks(title="RhymeLM Dashboard", css=dark_css, theme=gr.themes.Base(
    primary_hue="blue", secondary_hue="purple", neutral_hue="gray",
    font=gr.themes.GoogleFont("JetBrains Mono"),
)) as app:

    gr.Markdown("# RhymeLM — Live Dashboard\n"
                "Character-level LSTM generating 16-bar rap verses with rhyme-aware decoding")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="I ", lines=1)
                artist = gr.Dropdown(["None"] + ARTISTS, label="Artist Style", value="None")
                rhyme_scheme = gr.Dropdown(["free", "AABB", "ABAB", "ABBA"], label="Rhyme Scheme", value="free")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p (Nucleus)")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
                seed_val = gr.Number(label="Seed (-1 = random)", value=-1)
                use_orig = gr.Checkbox(label="Originality Filter (block training data copying)", value=True)
                gen_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                verse_out = gr.Textbox(label="Generated Verse", lines=14, elem_classes=["verse-output"])
                metrics_out = gr.Markdown(label="Metrics")

        gen_btn.click(generate_tab,
                      [prompt, artist, rhyme_scheme, temperature, top_p, rep_penalty, bars, seed_val, use_orig],
                      [verse_out, metrics_out])

    with gr.Tab("Artist Comparison"):
        with gr.Row():
            art_select = gr.CheckboxGroup(ARTISTS, label="Select Artists", value=ARTISTS[:4])
            art_temp = gr.Slider(0.3, 1.2, value=0.75, step=0.05, label="Temperature")
            art_scheme = gr.Dropdown(["free", "AABB", "ABAB"], value="AABB", label="Rhyme Scheme")
            art_samples = gr.Slider(2, 10, value=3, step=1, label="Samples per Artist")
        art_btn = gr.Button("Compare Artists", variant="primary")
        with gr.Row():
            radar_out = gr.Image(label="Radar Chart")
            bar_out = gr.Image(label="Bar Chart")
        art_btn.click(artist_comparison_tab, [art_select, art_temp, art_scheme, art_samples], [radar_out, bar_out])

    with gr.Tab("Temperature Explorer"):
        with gr.Row():
            temp_prompt = gr.Textbox(label="Prompt", value="I ", lines=1)
            temp_samples = gr.Slider(2, 10, value=3, step=1, label="Samples per Temp")
        temp_btn = gr.Button("Run Sweep", variant="primary")
        temp_plot = gr.Image(label="Temperature vs Quality")
        temp_btn.click(temperature_explorer_tab, [temp_prompt, temp_samples], temp_plot)

    with gr.Tab("Model Internals"):
        gr.Markdown("Explore what the LSTM learned internally: character embeddings, neuron specialization, and emergent representations.")
        internals_btn = gr.Button("Analyze Model", variant="primary")
        with gr.Row():
            embed_out = gr.Image(label="Embedding Space (t-SNE)")
        with gr.Row():
            neuron_out = gr.Image(label="Neuron Selectivity")
        with gr.Row():
            probe_out = gr.Image(label="Linear Probing")
        internals_btn.click(internals_tab, [], [embed_out, neuron_out, probe_out])


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
