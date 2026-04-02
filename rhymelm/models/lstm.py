"""LSTM-based character language model."""

import torch
import torch.nn as nn

from rhymelm.models.base import RhymeLMBase


class RhymeLM(RhymeLMBase):
    """
    LSTM backbone for character-level language modeling.

    Architecture: Embedding → LSTM → Dropout → Linear
    The LSTM captures sequential dependencies; its hidden state carries
    context across the sequence, making it naturally suited for modeling
    the flow and rhythm of verse.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.dropout(self.embed(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(emb)
        else:
            lstm_out, hidden = self.lstm(emb, hidden)

        logits = self.fc(self.dropout(lstm_out))
        return logits, hidden

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
