"""RhymeLM Live Dashboard — generate verses, compare artists, fine-tune on your own lyrics."""

import io
import os
import random
import time

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
from rhymelm.generation.originality import OriginalityFilter
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.data.corpus import load_lyrics, extract_verses, load_dictionary, build_dual_corpus, clean_user_lyrics
from rhymelm.data.phonemes import get_cmu_dict, get_rhyme_suffix, line_syllable_count
from rhymelm.data.lyrics_fetcher import fetch_artist_lyrics, build_training_corpus
from rhymelm.utils import seed_everything

# ── Resolve checkpoint ──
import glob
CKPT_CANDIDATES = [
    "checkpoints/rhymelm_transformer_bpe_final.pt",
    "checkpoints/rhymelm_original_final.pt",
    "checkpoints/rhymelm_lstm_final.pt",
]
ckpt_files = glob.glob("checkpoints/*final*.pt")
CHECKPOINT = ckpt_files[0] if ckpt_files else CKPT_CANDIDATES[-1]
CSV_PATH = "lyrics_cleaned.csv" if os.path.exists("lyrics_cleaned.csv") else "lyrics_raw.csv"
LYRICS_COL = "lyrics_clean" if "cleaned" in CSV_PATH else "artist_verses"
DEVICE = torch.device("cpu")

print(f"Loading model from {CHECKPOINT}...")
MODEL, TOKENIZER, CKPT = load_model(CHECKPOINT, DEVICE)
STARTERS = get_artist_starters(CSV_PATH, lyrics_column=LYRICS_COL)
CMU = get_cmu_dict()
ARTISTS = sorted(STARTERS.keys())

# Originality filter
print("Building originality filter...")
_lyrics = load_lyrics(CSV_PATH, LYRICS_COL)
_verses = extract_verses(_lyrics)
_corpus = "\n\n".join(_verses)
ORIG_FILTER = OriginalityFilter(_corpus, ngram_size=25)

# Fine-tuned model storage
FINETUNED_MODEL = None
FINETUNED_TOKENIZER = None

# Style
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "figure.dpi": 120, "font.family": "monospace",
})
PAL = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ff7b72",
       "#79c0ff", "#ffa657", "#56d364", "#bc8cff", "#ffd700"]


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def verse_to_html(verse: str) -> str:
    """Render verse as HTML with color-coded rhymes and syllable counts."""
    lines = [l for l in verse.strip().split("\n") if l.strip()]
    if not lines:
        return "<pre>No output</pre>"

    # Find rhyme groups
    endings = []
    for line in lines:
        words = line.split()
        suffix = get_rhyme_suffix(words[-1], CMU) if words else None
        endings.append(suffix)

    suffix_to_color = {}
    color_idx = 0
    for i, s in enumerate(endings):
        if s is None:
            continue
        if s not in suffix_to_color:
            # Only color if another line shares this suffix
            if endings.count(s) > 1:
                suffix_to_color[s] = PAL[color_idx % len(PAL)]
                color_idx += 1

    html_lines = []
    for i, line in enumerate(lines):
        words = line.split()
        syl = line_syllable_count(line, CMU)
        suffix = endings[i]
        color = suffix_to_color.get(suffix, "#c9d1d9")

        # Highlight last word if it rhymes
        if suffix and suffix in suffix_to_color and words:
            last_word = words[-1]
            rest = " ".join(words[:-1])
            line_html = (
                f'<span style="color:#8b949e">{i+1:>2}. </span>'
                f'<span style="color:#c9d1d9">{_esc(rest)} </span>'
                f'<span style="color:{color};font-weight:bold">{_esc(last_word)}</span>'
                f'<span style="color:#484f58;float:right">{syl} syl</span>'
            )
        else:
            line_html = (
                f'<span style="color:#8b949e">{i+1:>2}. </span>'
                f'<span style="color:#c9d1d9">{_esc(line)}</span>'
                f'<span style="color:#484f58;float:right">{syl} syl</span>'
            )
        html_lines.append(line_html)

    body = "<br>".join(html_lines)
    return f'<div style="font-family:monospace;font-size:14px;line-height:1.8;background:#0d1117;padding:16px;border-radius:8px;border:1px solid #30363d">{body}</div>'


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ═══════════════════════════════════════════════════════════════
# TAB 1: GENERATE
# ═══════════════════════════════════════════════════════════════
def generate_tab(prompt, artist, rhyme_scheme, temperature, top_p, rep_penalty, bars, seed_val, use_orig):
    if seed_val >= 0:
        seed_everything(int(seed_val))

    of = ORIG_FILTER if use_orig else None

    # Default to AABB when artist is selected and scheme is free
    effective_scheme = rhyme_scheme
    if artist and artist != "None" and rhyme_scheme == "free":
        effective_scheme = "AABB"

    if artist and artist != "None":
        verse = generate_artist_verse(
            MODEL, TOKENIZER, DEVICE, STARTERS, artist=artist,
            num_bars=int(bars), temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty,
            rhyme_scheme=effective_scheme if effective_scheme != "free" else None,
            originality_filter=of,
        )
    elif effective_scheme and effective_scheme != "free":
        verse = generate_rhyming_verse(
            MODEL, TOKENIZER, DEVICE, prompt=prompt or " ",
            num_bars=int(bars), temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty, rhyme_scheme=effective_scheme,
            originality_filter=of,
        )
    else:
        verse = generate_verse(
            MODEL, TOKENIZER, DEVICE, prompt=prompt or " ",
            num_bars=int(bars), temperature=temperature, top_p=top_p,
            repetition_penalty=rep_penalty, originality_filter=of,
        )

    metrics = evaluate_verse(verse)
    lines = [l.strip() for l in verse.strip().split("\n") if l.strip()]
    copied = sum(1 for l in lines if ORIG_FILTER.is_line_copied_fast(l))
    orig_pct = 1 - (copied / max(len(lines), 1))

    metrics_md = f"""| Metric | Value |
|---|---|
| **Originality** | **{orig_pct:.0%}** ({copied}/{len(lines)} lines from training) |
| Rhyme Density | **{metrics['rhyme_density']:.2%}** |
| Distinct-1 | {metrics['distinct_1']:.3f} |
| Distinct-2 | {metrics['distinct_2']:.3f} |
| Unique Lines | {metrics['unique_line_ratio']:.2%} |
| Lines | {metrics['num_lines']} |"""

    return verse_to_html(verse), verse, metrics_md


# ═══════════════════════════════════════════════════════════════
# TAB 2: ARTIST COMPARISON
# ═══════════════════════════════════════════════════════════════
def artist_comparison_tab(selected_artists, temperature, num_verses):
    if not selected_artists:
        selected_artists = ARTISTS[:4]

    results = {}
    for art in selected_artists:
        art_metrics = []
        for s in range(int(num_verses)):
            seed_everything(s)
            verse = generate_artist_verse(
                MODEL, TOKENIZER, DEVICE, STARTERS, artist=art,
                num_bars=8, temperature=temperature, top_p=0.95,
                repetition_penalty=1.15, originality_filter=ORIG_FILTER,
            )
            art_metrics.append(evaluate_verse(verse))
        results[art] = art_metrics

    # Radar chart
    metric_keys = ["rhyme_density", "distinct_1", "distinct_2", "unique_line_ratio"]
    labels = ["Rhyme", "Dist-1", "Dist-2", "Unique"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b22")
    for idx, art in enumerate(selected_artists):
        means = [np.mean([m[k] for m in results[art]]) for k in metric_keys] + \
                [np.mean([m[metric_keys[0]] for m in results[art]])]
        ax.plot(angles, means, "o-", linewidth=2.5, label=art, color=PAL[idx % len(PAL)], markersize=6)
        ax.fill(angles, means, alpha=0.1, color=PAL[idx % len(PAL)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color="#c9d1d9")
    ax.set_ylim(0, 1.1)
    ax.spines["polar"].set_color("#30363d")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title(f"Quality Metrics ({int(num_verses)} samples/artist)", fontsize=14, fontweight="bold", pad=25, color="#c9d1d9")
    return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════
# TAB 3: YOUR STYLE
# ═══════════════════════════════════════════════════════════════
def preview_lyrics(text, file):
    """Clean and preview uploaded lyrics."""
    if file is not None:
        with open(file.name) as f:
            text = f.read()
    if not text or not text.strip():
        return "No lyrics provided", ""

    cleaned = clean_user_lyrics(text)
    lines = [l for l in cleaned.splitlines() if l.strip()]
    stats = f"**{len(lines)} lines** | {len(cleaned):,} characters | {len(set(lines))} unique lines"
    return cleaned, stats


def fetch_lyrics_for_style(artist_name, genius_token):
    """Auto-fetch lyrics for an artist to use as training data."""
    if not artist_name:
        return "Enter an artist name", ""

    token = genius_token if genius_token else None
    songs = fetch_artist_lyrics(artist_name, max_songs=30, genius_token=token)

    if not songs:
        return f"Could not find lyrics for '{artist_name}'. Try providing a Genius API token.", ""

    all_lyrics = "\n\n".join(s["lyrics"] for s in songs)
    cleaned = clean_user_lyrics(all_lyrics)
    lines = [l for l in cleaned.splitlines() if l.strip()]
    stats = f"**{len(songs)} songs** | **{len(lines)} lines** | {len(cleaned):,} chars"
    return cleaned, stats


def finetune_on_lyrics(lyrics_text, use_lora, num_steps, progress=gr.Progress()):
    """Fine-tune the model on user lyrics."""
    global FINETUNED_MODEL, FINETUNED_TOKENIZER

    if not lyrics_text or len(lyrics_text.strip()) < 200:
        return "Need at least 200 characters of lyrics to fine-tune."

    from rhymelm.config import FinetuneConfig
    from rhymelm.training.finetune import prepare_user_data, finetune as do_finetune
    from rhymelm.data.dataset import create_dataloaders
    import copy

    ft_config = FinetuneConfig(
        use_lora=use_lora,
        lora_rank=8,
        num_steps=int(num_steps),
        learning_rate=3e-5 if use_lora else 1e-4,
        eval_interval=max(50, int(num_steps) // 10),
        early_stopping_patience=5,
    )

    # Prepare user data
    progress(0.1, desc="Preparing data...")
    user_train, user_val = prepare_user_data(
        lyrics_text, TOKENIZER,
        block_size=min(128, len(TOKENIZER.encode(lyrics_text[:500])) if hasattr(TOKENIZER, 'encode') else 128),
        batch_size=8,
    )

    # Clone model for fine-tuning
    progress(0.2, desc="Setting up model...")
    ft_model = copy.deepcopy(MODEL)
    ft_model.to(DEVICE)

    # Fine-tune
    progress(0.3, desc=f"Fine-tuning ({'LoRA' if use_lora else 'Full'})...")
    history = do_finetune(
        ft_model, user_train, user_val,
        base_train_loader=None,
        config=ft_config,
        device=DEVICE,
    )

    FINETUNED_MODEL = ft_model
    FINETUNED_TOKENIZER = TOKENIZER

    progress(1.0, desc="Done!")

    # Build loss curve
    if history["steps"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["steps"], history["train_loss"], "o-", label="Train", color=PAL[0], linewidth=2)
        ax.plot(history["steps"], history["val_loss"], "o-", label="Val", color=PAL[1], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Fine-Tuning Progress", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig_to_image(fig)

    return "Fine-tuning complete (no eval steps recorded)"


def generate_your_style(prompt, bars, temperature, top_p):
    """Generate using the fine-tuned model."""
    model = FINETUNED_MODEL if FINETUNED_MODEL is not None else MODEL
    tokenizer = FINETUNED_TOKENIZER if FINETUNED_TOKENIZER is not None else TOKENIZER

    verse = generate_verse(
        model, tokenizer, DEVICE,
        prompt=prompt or " ", num_bars=int(bars),
        temperature=temperature, top_p=top_p,
        repetition_penalty=1.15,
    )
    metrics = evaluate_verse(verse)
    is_finetuned = FINETUNED_MODEL is not None
    source = "Fine-tuned model" if is_finetuned else "Base model (no fine-tune yet)"

    return verse_to_html(verse), verse, f"**{source}** | rhyme: {metrics['rhyme_density']:.2%} | distinct-2: {metrics['distinct_2']:.3f}"


def compare_base_vs_finetuned(prompt, bars, temperature, seed_val):
    """Side-by-side: base model vs fine-tuned."""
    if FINETUNED_MODEL is None:
        return "Fine-tune first!", "", "Fine-tune first!", ""

    seed_everything(int(seed_val))
    base_verse = generate_verse(
        MODEL, TOKENIZER, DEVICE, prompt=prompt or " ",
        num_bars=int(bars), temperature=temperature, top_p=0.95,
        repetition_penalty=1.15,
    )

    seed_everything(int(seed_val))
    ft_verse = generate_verse(
        FINETUNED_MODEL, FINETUNED_TOKENIZER or TOKENIZER, DEVICE,
        prompt=prompt or " ", num_bars=int(bars),
        temperature=temperature, top_p=0.95,
        repetition_penalty=1.15,
    )

    bm = evaluate_verse(base_verse)
    fm = evaluate_verse(ft_verse)

    return (
        verse_to_html(base_verse),
        f"rhyme: {bm['rhyme_density']:.2%} | dist-2: {bm['distinct_2']:.3f}",
        verse_to_html(ft_verse),
        f"rhyme: {fm['rhyme_density']:.2%} | dist-2: {fm['distinct_2']:.3f}",
    )


# ═══════════════════════════════════════════════════════════════
# TAB 4: MODEL INTERNALS
# ═══════════════════════════════════════════════════════════════
def internals_tab():
    from sklearn.manifold import TSNE
    from rhymelm.evaluation.interpretability import extract_hidden_states, build_word_boundary_labels, train_probe
    from collections import Counter

    # Embedding space (works for both char and BPE)
    weights = MODEL.token_embed.weight.detach().cpu().numpy() if hasattr(MODEL, "token_embed") else MODEL.embed.weight.detach().cpu().numpy()
    itos = TOKENIZER.itos
    chars = [itos.get(i, "?") for i in range(min(len(itos), weights.shape[0]))]
    mask = [len(c.strip()) > 0 and c.isprintable() for c in chars]
    fw = weights[mask]
    fc = [c for c, m in zip(chars, mask) if m]

    if len(fc) > 5:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fc) - 1))
        coords = tsne.fit_transform(fw)

        fig1, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(coords[:, 0], coords[:, 1], c=PAL[0], s=40, alpha=0.6, edgecolors="white", linewidth=0.3)
        for i, ch in enumerate(fc):
            if len(ch) <= 3:
                ax.annotate(ch, (coords[i, 0], coords[i, 1]), fontsize=7, ha="center", va="bottom",
                            color="white", path_effects=[pe.withStroke(linewidth=1.5, foreground="#0d1117")])
        ax.set_title("Token Embedding Space (t-SNE)", fontsize=15, fontweight="bold", pad=12)
        ax.grid(True, alpha=0.15)
    else:
        fig1, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Not enough tokens to visualize", ha="center", va="center", fontsize=14)

    embed_img = fig_to_image(fig1)

    # Model info
    total_params = sum(p.numel() for p in MODEL.parameters())
    arch = CKPT["config"]["model"].get("arch", "unknown")
    info = f"**{arch}** | {total_params:,} params | checkpoint step {CKPT.get('step', '?')}"

    return embed_img, info


# ═══════════════════════════════════════════════════════════════
# BUILD GRADIO APP
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="RhymeLM Dashboard") as app:
    gr.Markdown("# RhymeLM — Verse Generation Machine\n"
                "Generate original rap verses in any artist's style, or fine-tune on your own lyrics")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="I ", lines=1)
                artist = gr.Dropdown(["None"] + ARTISTS, label="Artist Style", value="None")
                rhyme_scheme = gr.Dropdown(["free", "AABB", "ABAB", "ABBA"], label="Rhyme Scheme", value="free")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")
                bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
                seed_val = gr.Number(label="Seed (-1=random)", value=-1)
                use_orig = gr.Checkbox(label="Originality Filter", value=True)
                gen_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                verse_html = gr.HTML(label="Verse")
                verse_raw = gr.Textbox(label="Raw Text", lines=10, visible=False)
                metrics_out = gr.Markdown()

        gen_btn.click(generate_tab,
                      [prompt, artist, rhyme_scheme, temperature, top_p, rep_penalty, bars, seed_val, use_orig],
                      [verse_html, verse_raw, metrics_out])

    with gr.Tab("Artist Comparison"):
        with gr.Row():
            art_select = gr.CheckboxGroup(ARTISTS, label="Artists", value=ARTISTS[:4])
            art_temp = gr.Slider(0.3, 1.2, value=0.75, step=0.05, label="Temperature")
            art_samples = gr.Slider(2, 10, value=3, step=1, label="Samples/Artist")
        art_btn = gr.Button("Compare", variant="primary")
        radar_out = gr.Image(label="Radar Chart")
        art_btn.click(artist_comparison_tab, [art_select, art_temp, art_samples], radar_out)

    with gr.Tab("Your Style"):
        gr.Markdown("### Upload your lyrics or fetch an artist's catalog, then fine-tune the model")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Option A: Paste/Upload Lyrics**")
                lyrics_input = gr.Textbox(label="Paste lyrics here", lines=10, placeholder="Paste your lyrics here...")
                lyrics_file = gr.File(label="Or upload .txt file", file_types=[".txt", ".csv"])
                preview_btn = gr.Button("Clean & Preview")

            with gr.Column():
                gr.Markdown("**Option B: Auto-Fetch Artist Lyrics**")
                fetch_artist = gr.Textbox(label="Artist Name", placeholder="e.g. Kendrick Lamar")
                genius_token = gr.Textbox(label="Genius API Token (optional)", type="password",
                                          placeholder="Leave blank to use existing dataset")
                fetch_btn = gr.Button("Fetch Lyrics", variant="secondary")

        with gr.Row():
            cleaned_text = gr.Textbox(label="Cleaned Lyrics (ready for training)", lines=8)
            data_stats = gr.Markdown()

        preview_btn.click(preview_lyrics, [lyrics_input, lyrics_file], [cleaned_text, data_stats])
        fetch_btn.click(fetch_lyrics_for_style, [fetch_artist, genius_token], [cleaned_text, data_stats])

        gr.Markdown("### Fine-Tune")
        with gr.Row():
            use_lora = gr.Checkbox(label="LoRA (fast, ~2 min)", value=True)
            ft_steps = gr.Slider(100, 2000, value=500, step=100, label="Steps")
        ft_btn = gr.Button("Fine-Tune Model", variant="primary")
        ft_output = gr.Image(label="Training Progress")
        ft_btn.click(finetune_on_lyrics, [cleaned_text, use_lora, ft_steps], ft_output)

        gr.Markdown("### Generate with Your Model")
        with gr.Row():
            ys_prompt = gr.Textbox(label="Prompt", value="I ", lines=1)
            ys_bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
            ys_temp = gr.Slider(0.3, 1.2, value=0.8, step=0.05, label="Temp")
            ys_top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.05, label="Top-p")
        ys_btn = gr.Button("Generate (Your Style)", variant="primary")
        ys_verse = gr.HTML()
        ys_raw = gr.Textbox(visible=False)
        ys_metrics = gr.Markdown()
        ys_btn.click(generate_your_style, [ys_prompt, ys_bars, ys_temp, ys_top_p], [ys_verse, ys_raw, ys_metrics])

        gr.Markdown("### Compare: Base vs Fine-Tuned")
        with gr.Row():
            cmp_prompt = gr.Textbox(label="Prompt", value="I ", lines=1)
            cmp_bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
            cmp_temp = gr.Slider(0.3, 1.2, value=0.8, step=0.05, label="Temp")
            cmp_seed = gr.Number(label="Seed", value=42)
        cmp_btn = gr.Button("Compare Side-by-Side", variant="secondary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Base Model**")
                cmp_base_verse = gr.HTML()
                cmp_base_metrics = gr.Markdown()
            with gr.Column():
                gr.Markdown("**Your Model**")
                cmp_ft_verse = gr.HTML()
                cmp_ft_metrics = gr.Markdown()
        cmp_btn.click(compare_base_vs_finetuned,
                      [cmp_prompt, cmp_bars, cmp_temp, cmp_seed],
                      [cmp_base_verse, cmp_base_metrics, cmp_ft_verse, cmp_ft_metrics])

    with gr.Tab("Model Internals"):
        gr.Markdown("Explore what the model learned: token embeddings, architecture info")
        int_btn = gr.Button("Analyze", variant="primary")
        int_embed = gr.Image(label="Embedding Space")
        int_info = gr.Markdown()
        int_btn.click(internals_tab, [], [int_embed, int_info])


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
