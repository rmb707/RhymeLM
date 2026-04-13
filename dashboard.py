"""RhymeLM Dashboard — RAG-based verse generation.

Instead of fine-tuning a massive model on limited data, the system retrieves
stylistically similar verses from an indexed corpus and uses them as in-context
examples for GPT-2 XL to generate new bars in the same style.
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
BASE_MODEL = "gpt2-xl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("RhymeLM RAG Dashboard")
print("=" * 60)

# Load index + generator once
if not Path(INDEX_PATH).exists():
    raise SystemExit(
        f"Index not found at {INDEX_PATH}. "
        f"Build it first with: python -m rhymelm.rag.index --csv lyrics_cleaned.csv"
    )

INDEX = VerseIndex.load(INDEX_PATH)
GENERATOR = RAGGenerator(INDEX, model_name=BASE_MODEL, device=DEVICE)

# Available artists from the index
ARTISTS = sorted(set(c.artist for c in INDEX.chunks))
print(f"Artists in index: {', '.join(ARTISTS)}")

STYLES = {
    "Balanced":  {"temperature": 0.85, "top_p": 0.92},
    "Sharp":     {"temperature": 0.7,  "top_p": 0.9},
    "Wild":      {"temperature": 1.0,  "top_p": 0.95},
    "Focused":   {"temperature": 0.6,  "top_p": 0.85},
}


def generate(artist, style, topic, num_bars):
    if not topic or not topic.strip():
        return "Enter a topic.", "", ""

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
        retrieved_md += f"**{i}. {chunk.artist} — {chunk.title}** (similarity: {score:.2f})\n\n"
        retrieved_md += f"```\n{preview}\n```\n\n"

    # Numbered output
    lines = result.verse.strip().split("\n")
    formatted = "\n".join(f"{i+1:>2}. {line}" for i, line in enumerate(lines))

    info = f"**{artist}** | Style: {style} | {len(lines)} bars | Retrieved {len(result.retrieved_chunks)} examples"

    return formatted, info, retrieved_md


with gr.Blocks(
    title="RhymeLM RAG",
    theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"),
) as app:
    gr.Markdown(
        "# RhymeLM — Retrieval-Augmented Verse Generation\n"
        "Pick an artist, give it a topic, and the system retrieves matching verses "
        "from the corpus to guide GPT-2 XL in writing new bars in that style.\n"
        "No fine-tuning — pure retrieval + in-context learning."
    )

    with gr.Row():
        with gr.Column(scale=1):
            artist = gr.Dropdown(ARTISTS, label="Artist", value=ARTISTS[0] if ARTISTS else None)
            style = gr.Dropdown(list(STYLES.keys()), label="Style", value="Balanced")
            bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
            topic = gr.Textbox(
                label="Topic",
                placeholder="e.g. coming up from nothing",
                lines=2,
            )
            gen_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            verse_out = gr.Textbox(label="Generated Verse", lines=12)
            info_out = gr.Markdown()

    with gr.Accordion("Retrieved Context (the RAG system's sources)", open=False):
        retrieved_out = gr.Markdown()

    gen_btn.click(generate, [artist, style, topic, bars], [verse_out, info_out, retrieved_out])

    with gr.Tab("How it works"):
        gr.Markdown(
            """
            ### Architecture

            This is a **Retrieval-Augmented Generation (RAG)** system:

            1. **Index phase** (offline): every verse in the lyrics corpus is split into
               4–16 line chunks, embedded using `sentence-transformers/all-MiniLM-L6-v2`
               (384-dim vectors), and stored in a FAISS inner-product index.

            2. **Query phase** (per request):
               - Your topic is embedded with the same encoder
               - FAISS returns the top-k most similar verses from the target artist
               - Those verses are injected into a prompt as few-shot examples
               - GPT-2 XL (1.5B params, no fine-tuning) completes the pattern

            3. **Generation**: the base model imitates the retrieved examples
               in-context, producing bars in the same vocabulary, flow, and
               emotional register as the real verses.

            ### Why RAG instead of fine-tuning?

            Fine-tuning a 1.5B model on ~500 songs overfits catastrophically — it
            memorizes chunks of training data without preserving the base model's
            language understanding. RAG sidesteps this entirely: the base model
            stays intact and we inject style at inference time.

            ### Components

            - **Encoder**: sentence-transformers/all-MiniLM-L6-v2 (22M params)
            - **Vector store**: FAISS IndexFlatIP (exact inner product)
            - **Generator**: GPT-2 XL (1.5B params, bfloat16)
            - **Index size**: ~4000 verse chunks across 11 artists
            """
        )


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
