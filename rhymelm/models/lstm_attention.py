"""LSTM + Self-Attention hybrid model.

Adds a single multi-head self-attention layer on top of LSTM output.
The LSTM provides strong local sequential modeling; attention adds the
ability to look back at distant tokens — like a rhyming word several
bars ago. This hybrid demonstrates that attention and recurrence are
complementary, not just replacements.
"""

import torch
import torch.nn as nn

from rhymelm.models.base import RhymeLMBase


class RhymeLMAttention(RhymeLMBase):
    """LSTM + self-attention hybrid for character language modeling.

    Architecture: Embedding → LSTM → Self-Attention (residual + LayerNorm) → Linear

    Returns attention weights alongside logits for visualization.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Self-attention over LSTM output
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def forward(self, x: torch.Tensor, hidden=None, return_attention=False):
        emb = self.dropout(self.embed(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(emb)
        else:
            lstm_out, hidden = self.lstm(emb, hidden)

        # Causal mask: prevent attending to future positions
        seq_len = lstm_out.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Self-attention with residual connection and layer norm
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            attn_mask=causal_mask,
            need_weights=return_attention,
        )
        out = self.attn_norm(lstm_out + attn_out)

        logits = self.fc(self.dropout(out))

        if return_attention:
            return logits, hidden, attn_weights
        return logits, hidden

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
