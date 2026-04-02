"""Decoder-only Transformer built from scratch.

Every component is hand-implemented to demonstrate understanding of the
architecture at the component level — not imported from nn.MultiheadAttention.

Architecture follows GPT-2 conventions:
- Pre-norm (LayerNorm before attention and FFN, not after)
- GELU activation in the feed-forward network
- Learned positional embeddings
- Causal (autoregressive) masking

The model supports KV-caching for efficient autoregressive generation:
without cache, each new token recomputes attention over all previous tokens (O(n²));
with cache, only the new token's queries attend to cached keys/values (O(n) per step).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase


class CausalSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention with causal masking.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Each head operates on a d_k = d_model / n_heads dimensional subspace.
    The causal mask ensures position i can only attend to positions <= i,
    which is required for autoregressive (left-to-right) generation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B, T, C) -> (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # KV-cache: append new K,V to cached values
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Scaled dot-product attention
        # (B, heads, T_q, d_k) @ (B, heads, d_k, T_kv) -> (B, heads, T_q, T_kv)
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale

        # Apply causal mask
        T_q, T_kv = q.size(2), k.size(2)
        # For cached generation, we only mask based on the query/key positions
        mask = self.causal_mask[:T_q, :T_kv] if kv_cache is None else None
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # (B, heads, T_q, T_kv) @ (B, heads, T_kv, d_k) -> (B, heads, T_q, d_k)
        out = attn_weights @ v

        # Concatenate heads: (B, heads, T, d_k) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        out = self.resid_dropout(self.out_proj(out))

        return out, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    FFN(x) = Linear(GELU(Linear(x)))

    GELU is used over ReLU following GPT-2: it provides a smooth
    approximation that allows small negative gradients, improving
    training stability in deep transformer stacks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm before attention and FFN.

    Pre-norm (vs post-norm) is the modern standard from GPT-2 onward.
    It trains more stably because the residual path carries unscaled
    gradients, avoiding the vanishing gradient problem in deep stacks.

    x → LN → Attention → + residual → LN → FFN → + residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        attn_out, attn_weights = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Attention is permutation-invariant — without position information,
    the model cannot distinguish "the cat sat" from "sat the cat".
    Sinusoidal encoding uses fixed sine/cosine functions at different
    frequencies, giving each position a unique signature that the model
    can learn to interpret as relative distance.
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0):
        return x + self.pe[:, offset : offset + x.size(1)]


class RhymeLMTransformer(RhymeLMBase):
    """Decoder-only transformer for character-level language modeling.

    Architecture:
        Token Embedding + Positional Encoding
        → N x TransformerBlock (pre-norm, causal self-attention, FFN)
        → LayerNorm
        → Linear head → vocab logits

    Supports two positional encoding modes:
        - 'sinusoidal': fixed sin/cos (Vaswani et al.)
        - 'learned': trainable embedding lookup (GPT-2 style)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)

        if pos_encoding == "sinusoidal":
            self.pos_embed = PositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.pos_encoding_type = pos_encoding
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Weight tying: share token embedding with output projection
        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, kv_caches: list[dict] | None = None, offset: int = 0):
        B, T = x.shape
        tok_emb = self.token_embed(x)

        if self.pos_encoding_type == "sinusoidal":
            x_emb = self.pos_embed(tok_emb, offset=offset)
        else:
            positions = torch.arange(offset, offset + T, device=x.device)
            x_emb = tok_emb + self.pos_embed(positions)

        x_emb = self.drop(x_emb)

        all_attn_weights = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x_emb, attn_w = block(x_emb, cache)
            all_attn_weights.append(attn_w)

        x_emb = self.ln_final(x_emb)
        logits = self.head(x_emb)

        return logits, all_attn_weights

    def init_state(self, batch_size: int, device: torch.device):
        """Return empty KV caches for each layer."""
        return [{} for _ in range(self.n_layers)]

    def generate_with_cache(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache for O(n) per step.

        Without cache: each new token recomputes attention over all positions → O(n²) total.
        With cache: previous keys/values are stored and reused → O(n) per step.
        """
        kv_caches = self.init_state(x.size(0), x.device)

        # Process prompt (prefill)
        logits, _ = self(x, kv_caches, offset=0)
        generated = [x]

        for step in range(max_new_tokens):
            # Only feed the last token; cached K,V handle context
            last_logits = logits[:, -1:, :] / max(temperature, 1e-8)

            if top_k > 0:
                top_k_val = min(top_k, last_logits.size(-1))
                threshold = torch.topk(last_logits, top_k_val, dim=-1).values[:, :, -1:]
                last_logits = last_logits.masked_fill(last_logits < threshold, float("-inf"))

            if top_p > 0.0:
                sorted_l, sorted_i = torch.sort(last_logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                mask = cum - F.softmax(sorted_l, dim=-1) >= top_p
                sorted_l[mask] = float("-inf")
                last_logits = sorted_l.scatter(-1, sorted_i, sorted_l)

            probs = F.softmax(last_logits.squeeze(1), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)

            offset = x.size(1) + step
            logits, _ = self(next_tok, kv_caches, offset=offset)

        return torch.cat(generated, dim=1)
