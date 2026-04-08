"""RhymeLM Dashboard — clean, simple verse generation."""

import os
import torch
import gradio as gr

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

HF_MODEL_DIR = "checkpoints/hf_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ──
MODEL = None
TOKENIZER = None


def load_hf_model():
    global MODEL, TOKENIZER
    if os.path.exists(HF_MODEL_DIR):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading HF model from {HF_MODEL_DIR}...")
        TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
        MODEL = AutoModelForCausalLM.from_pretrained(HF_MODEL_DIR, torch_dtype=torch.bfloat16).to(DEVICE)
        MODEL.eval()
        params = sum(p.numel() for p in MODEL.parameters())
        print(f"Loaded: {params/1e6:.0f}M params on {DEVICE}")
        return True
    return False


HF_LOADED = load_hf_model()

ARTISTS = ["Eminem", "Drake", "Kendrick Lamar", "2Pac", "Nas",
           "J. Cole", "Nicki Minaj", "Future", "Dave", "Skepta", "Rapsody"]

STYLES = {
    "Hard": {"temperature": 0.8, "top_p": 0.9, "rep_penalty": 1.2},
    "Smooth": {"temperature": 0.7, "top_p": 0.95, "rep_penalty": 1.1},
    "Creative": {"temperature": 0.95, "top_p": 0.95, "rep_penalty": 1.15},
    "Tight": {"temperature": 0.6, "top_p": 0.85, "rep_penalty": 1.25},
    "Loose": {"temperature": 1.0, "top_p": 0.98, "rep_penalty": 1.05},
}


def generate(artist, style, bars, topic):
    if MODEL is None or TOKENIZER is None:
        return "Model not loaded. Train first: `python -m rhymelm.training.hf_finetune`"

    params = STYLES.get(style, STYLES["Smooth"])
    prompt = f"<|artist|>{artist}<|verse|>\n"
    if topic and topic.strip():
        prompt = f"<|artist|>{artist}<|verse|>\n{topic.strip()}\n"

    inputs = TOKENIZER.encode(prompt, return_tensors="pt").to(DEVICE)
    end_id = TOKENIZER.encode("<|end|>")[0] if "<|end|>" in TOKENIZER.get_vocab() else TOKENIZER.eos_token_id

    with torch.no_grad():
        out = MODEL.generate(
            inputs,
            max_new_tokens=int(bars) * 30,
            temperature=params["temperature"],
            top_p=params["top_p"],
            do_sample=True,
            repetition_penalty=params["rep_penalty"],
            eos_token_id=end_id,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    text = TOKENIZER.decode(out[0], skip_special_tokens=False)
    if "<|verse|>" in text:
        text = text.split("<|verse|>")[-1]
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()][:int(bars)]
    verse = "\n".join(lines)

    return verse


def generate_and_format(artist, style, bars, topic):
    verse = generate(artist, style, bars, topic)
    lines = verse.strip().split("\n")

    # Simple numbered format
    formatted = ""
    for i, line in enumerate(lines):
        formatted += f"{i+1:>2}. {line}\n"

    info = f"**{artist}** | Style: {style} | {len(lines)} bars"
    return formatted, info


# ── Build UI ──
with gr.Blocks(
    title="RhymeLM",
    theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"),
) as app:
    gr.Markdown(
        "# RhymeLM\n"
        "Generate original rap verses in any artist's style"
    )

    with gr.Row():
        with gr.Column(scale=1):
            artist = gr.Dropdown(ARTISTS, label="Artist", value="Eminem")
            style = gr.Dropdown(list(STYLES.keys()), label="Style", value="Smooth")
            bars = gr.Slider(4, 16, value=8, step=1, label="Bars")
            topic = gr.Textbox(label="Topic (optional)", placeholder="e.g. coming up from nothing", lines=1)
            gen_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            verse_out = gr.Textbox(label="Verse", lines=18)
            info_out = gr.Markdown()

    gen_btn.click(generate_and_format, [artist, style, bars, topic], [verse_out, info_out])

    with gr.Tab("Your Style"):
        gr.Markdown("### Paste your own lyrics to fine-tune the model on your style")
        lyrics_in = gr.Textbox(label="Your Lyrics", lines=12, placeholder="Paste your bars here...")
        with gr.Row():
            ft_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            ft_rank = gr.Slider(8, 64, value=32, step=8, label="LoRA Rank")
        ft_btn = gr.Button("Fine-Tune on Your Lyrics", variant="primary")
        ft_status = gr.Markdown()

        def finetune_user_lyrics(lyrics_text, epochs, rank, progress=gr.Progress()):
            if not lyrics_text or len(lyrics_text.strip()) < 200:
                return "Need at least 200 characters of lyrics."

            progress(0.1, desc="Preparing...")

            # Write temp file
            import tempfile, pandas as pd
            lines = [l.strip() for l in lyrics_text.strip().split("\n") if l.strip()]
            df = pd.DataFrame({"artist": ["You"] * len(lines), "lyrics_clean": ["\n".join(lines)]})
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
            df.to_csv(tmp.name, index=False)

            progress(0.3, desc="Fine-tuning...")
            from rhymelm.training.hf_finetune import finetune_hf
            model, tokenizer = finetune_hf(
                csv_path=tmp.name, lyrics_col="lyrics_clean",
                num_epochs=int(epochs), lora_rank=int(rank),
            )

            global MODEL, TOKENIZER
            MODEL = model
            TOKENIZER = tokenizer
            progress(1.0, desc="Done!")
            return f"Fine-tuned on {len(lines)} lines. Generate with artist='You'"

        ft_btn.click(finetune_user_lyrics, [lyrics_in, ft_epochs, ft_rank], ft_status)

    with gr.Tab("About"):
        gr.Markdown("""
### RhymeLM
Character-level + GPT-2 XL verse generation with LoRA fine-tuning.

**Architecture:**
- Base: GPT-2 XL (1.5B parameters)
- Fine-tuned with LoRA on rap lyrics from 11 artists
- Artist conditioning via special tokens: `<|artist|>Name<|verse|>`

**How it works:**
1. Select an artist — the model learned their vocabulary, flow, and style from training
2. Pick a style preset (controls creativity vs consistency)
3. Optionally add a topic to steer the verse
4. Hit Generate

**Your Style tab:** Paste your own lyrics, fine-tune the model on your patterns, then generate as "You".
        """)


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
