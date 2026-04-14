"""RhymeLM Dashboard — RAG-based verse generation with per-artist style profiles.

Pipeline (no fine-tuning required):
1. Per-artist style profiles extracted from the lyrics corpus offline (rhyme
   scheme, signature vocabulary, syllable density, contraction rate)
2. FAISS-indexed verse chunks tagged with their own rhyme scheme
3. Retrieval re-ranks chunks to prefer the artist's dominant scheme
4. Prompt injects the style profile so the model knows what it's imitating
5. RhymeBiasLogitsProcessor nudges generation toward rhyming completions
"""

import os
from pathlib import Path

import torch
import gradio as gr

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from rhymelm.rag import VerseIndex
from rhymelm.rag.generator import RAGGenerator


INDEX_PATH = "rag_index"
PROFILES_PATH = "artist_profiles.json"
BASE_MODEL = "gpt2-xl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("RhymeLM RAG Dashboard")
print("=" * 60)

if not Path(INDEX_PATH).exists():
    raise SystemExit(
        f"Index not found at {INDEX_PATH}. "
        f"Build it first with: python -m rhymelm.rag.index --csv lyrics_cleaned.csv"
    )

INDEX = VerseIndex.load(INDEX_PATH)
GENERATOR = RAGGenerator(
    INDEX,
    model_name=BASE_MODEL,
    device=DEVICE,
    profiles_path=PROFILES_PATH,
)

ARTISTS = sorted(set(c.artist for c in INDEX.chunks))
print(f"Artists in index: {', '.join(ARTISTS)}")

STYLES = {
    "Balanced":  {"temperature": 0.85, "top_p": 0.92},
    "Sharp":     {"temperature": 0.7,  "top_p": 0.9},
    "Wild":      {"temperature": 1.0,  "top_p": 0.95},
    "Focused":   {"temperature": 0.6,  "top_p": 0.85},
}


SCHEME_DESCRIPTIONS = {
    "AABB": "AABB couplets — lines 1+2 rhyme, then 3+4 rhyme",
    "ABAB": "ABAB alternating — line 1 with line 3, line 2 with line 4",
    "ABBA": "ABBA enclosed — outer pair rhymes, inner pair rhymes",
    "AAAA": "AAAA — same rhyme held across every bar (multisyllabic chains)",
    "free": "free verse — no fixed pattern",
    "irregular": "no consistent pattern detected",
}


def format_profile_card(artist: str) -> str:
    """Format a profile card as Markdown for the dashboard."""
    profile = GENERATOR.get_profile(artist)
    if not profile:
        return f"_No style profile available for {artist}_"

    scheme = profile.get("dominant_scheme", "free")
    conf = profile.get("scheme_confidence_named", 0)
    avg_syl = profile.get("avg_syllables", 0)
    syl_std = profile.get("syllable_stddev", 0)
    density = profile.get("rhyme_density", 0)
    contraction = profile.get("contraction_rate", 0)
    sig_words = profile.get("signature_words", [])[:8]

    scheme_desc = SCHEME_DESCRIPTIONS.get(scheme, scheme)

    md = f"### Studied profile: {artist}\n\n"
    md += f"- **Rhyme scheme:** `{scheme}` — {scheme_desc} ({conf:.0%} of clean patterns)\n"
    md += f"- **Bar length:** {avg_syl:.1f} syllables (±{syl_std:.1f})\n"
    md += f"- **Rhyme density:** {density:.0%} of adjacent bars rhyme\n"
    md += f"- **Contraction rate:** {contraction:.0%} (casual / AAVE intensity)\n"
    if sig_words:
        md += f"- **Signature words:** {', '.join(f'`{w}`' for w in sig_words)}\n"
    return md


def generate(artist, style, topic, num_bars):
    if not topic or not topic.strip():
        return "Enter a topic.", "", "", format_profile_card(artist)

    params = STYLES.get(style, STYLES["Balanced"])
    result = GENERATOR.generate(
        topic=topic.strip(),
        artist=artist,
        num_bars=int(num_bars),
        temperature=params["temperature"],
        top_p=params["top_p"],
    )

    # Format retrieved context for display
    retrieved_md = "### Retrieved Examples (from corpus)\n\n"
    for i, (chunk, score) in enumerate(zip(result.retrieved_chunks, result.retrieval_scores), 1):
        preview = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
        scheme_tag = f" `[{chunk.rhyme_scheme}]`" if getattr(chunk, "rhyme_scheme", None) else ""
        retrieved_md += f"**{i}. {chunk.artist} — {chunk.title}**{scheme_tag} (similarity: {score:.2f})\n\n"
        retrieved_md += f"```\n{preview}\n```\n\n"

    # Numbered output
    lines = result.verse.strip().split("\n")
    formatted = "\n".join(f"{i+1:>2}. {line}" for i, line in enumerate(lines))

    scheme_used = result.profile.get("dominant_scheme", "free") if result.profile else "free"
    info = (
        f"**{artist}** | Style: {style} | Scheme: `{scheme_used}` | "
        f"{len(lines)} bars | Retrieved {len(result.retrieved_chunks)} examples"
    )

    return formatted, info, retrieved_md, format_profile_card(artist)


def update_profile_card(artist):
    return format_profile_card(artist)


with gr.Blocks(
    title="RhymeLM RAG",
    theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"),
) as app:
    gr.Markdown(
        "# RhymeLM — Retrieval-Augmented Verse Generation\n"
        "Pick an artist, give it a topic. The system retrieves matching verses "
        "from the corpus and uses the artist's studied style profile (dominant rhyme "
        "scheme, signature vocabulary) to guide GPT-2 XL.\n"
        "**No fine-tuning** — pure retrieval + in-context learning + rhyme-aware decoding."
    )

    with gr.Row():
        with gr.Column(scale=1):
            artist = gr.Dropdown(ARTISTS, label="Artist", value=ARTISTS[0] if ARTISTS else None)
            style = gr.Dropdown(list(STYLES.keys()), label="Sampling style", value="Balanced")
            bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
            topic = gr.Textbox(
                label="Topic",
                placeholder="e.g. coming up from nothing",
                lines=2,
            )
            gen_btn = gr.Button("Generate", variant="primary", size="lg")

            profile_card = gr.Markdown(
                format_profile_card(ARTISTS[0]) if ARTISTS else "",
                label="Studied profile",
            )

        with gr.Column(scale=2):
            verse_out = gr.Textbox(label="Generated Verse", lines=12)
            info_out = gr.Markdown()

    with gr.Accordion("Retrieved Context (RAG sources)", open=False):
        retrieved_out = gr.Markdown()

    # When the artist changes, refresh the profile card
    artist.change(update_profile_card, [artist], [profile_card])

    gen_btn.click(
        generate,
        [artist, style, topic, bars],
        [verse_out, info_out, retrieved_out, profile_card],
    )

    with gr.Tab("How it works"):
        gr.Markdown(
            """
            ### Architecture

            This is a **Retrieval-Augmented Generation (RAG)** system with
            per-artist style profiles and rhyme-aware decoding.

            #### 1. Offline phases (run once)

            **Profiler** (`rhymelm/rag/profiler.py`):
            - Slides a 4-line window across every artist's verses in the corpus
            - Classifies each window as AABB / ABAB / ABBA / AAAA / irregular by
              comparing CMU-dict rhyme suffixes of line endings
            - Tallies per-artist scheme distribution to identify dominant pattern
            - Computes signature vocabulary via TF-IDF (per-artist vs corpus)
            - Saves to `artist_profiles.json`

            **Index** (`rhymelm/rag/index.py`):
            - Splits each verse into 4–16 line chunks with 50% overlap
            - Tags each chunk with its own rhyme scheme classification
            - Embeds with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
            - Stores in FAISS `IndexFlatIP` (exact cosine via normalized vectors)

            #### 2. Per-request pipeline

            1. **Profile lookup** — fetch the artist's dominant scheme + signature words
            2. **Retrieval** — FAISS top-k filtered by artist, then re-ranked to
               prefer chunks whose own scheme matches the artist's dominant pattern
            3. **MMR diversity** — drop near-duplicate chunks
            4. **Prompt assembly** — inject profile metadata, list 3 retrieved verses,
               instruct the model to write a new verse in the target scheme
            5. **Generation** — GPT-2 XL with `RhymeBiasLogitsProcessor`:
               - Tracks completed lines + which scheme group each belongs to
               - Establishes target rhyme suffix per group from the first line
               - At line endings, additively boosts logits for tokens that
                 complete a word matching the target suffix
               - Soft bias (additive) — coherence wins ties

            #### Why this beats fine-tuning

            Fine-tuning a 1.5B model on ~500 songs overfits catastrophically — it
            memorizes chunks of training data while wrecking the base model's
            language ability. RAG keeps GPT-2 XL intact and injects style at
            inference time through three reinforcing layers.

            #### Components

            - **Encoder:** sentence-transformers/all-MiniLM-L6-v2 (22M params)
            - **Vector store:** FAISS IndexFlatIP, ~4000 verse chunks
            - **Generator:** GPT-2 XL (1.5B params, bfloat16) — never modified
            - **Profiler:** CMU pronouncing dict + TF-IDF
            - **Logits processor:** custom `RhymeBiasLogitsProcessor`
            """
        )


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
