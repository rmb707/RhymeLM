"""LSTM with phoneme auxiliary head for multi-task learning.

The auxiliary head predicts phoneme sequences at word boundaries, forcing
the LSTM hidden state to encode pronunciation information. This is the
same principle behind auxiliary objectives in BERT (NSP) and multi-task
learning generally: shared representations learn richer features when
trained on complementary objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase


class RhymeLMPhoneme(RhymeLMBase):
    """LSTM + phoneme auxiliary head for rhyme-aware character modeling.

    Architecture:
        Embedding → LSTM → Dropout → [char head, phoneme head]

    The phoneme head operates at word-boundary positions only,
    predicting the phoneme sequence of the word that just completed.
    """

    def __init__(
        self,
        vocab_size: int,
        phoneme_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_phonemes: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.phoneme_vocab_size = phoneme_vocab_size
        self.max_phonemes = max_phonemes

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        # Primary head: next character prediction
        self.char_head = nn.Linear(hidden_dim, vocab_size)

        # Auxiliary head: phoneme sequence prediction at word boundaries
        # Projects hidden state to a sequence of phoneme predictions
        self.phoneme_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_phonemes * phoneme_vocab_size),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.dropout(self.embed(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(emb)
        else:
            lstm_out, hidden = self.lstm(emb, hidden)

        lstm_out = self.dropout(lstm_out)
        char_logits = self.char_head(lstm_out)

        return char_logits, hidden

    def phoneme_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict phoneme sequences from hidden states at word boundaries.

        Args:
            hidden_states: (num_boundaries, hidden_dim)

        Returns:
            phoneme_logits: (num_boundaries, max_phonemes, phoneme_vocab_size)
        """
        raw = self.phoneme_head(hidden_states)
        return raw.view(-1, self.max_phonemes, self.phoneme_vocab_size)

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
