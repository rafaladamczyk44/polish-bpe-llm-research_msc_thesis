import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PositionalEncoding, TransformerBlock

class LanguageModel(nn.Module):
    """
    Language model which will test the thesis
    """
    def __init__(self,
                 vocab_size,
                 d_model=768,
                 num_heads=12,
                 num_layers=6,
                 context_size=512,
                 d_ff=3072,
                 dropout=0.1,
                 max_len=5000):
        super(LanguageModel, self).__init__()

        self.d_model = d_model
        self.context_size = context_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_size, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Final projection onto vocab
        self.output = nn.Linear(d_model, vocab_size)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with Xavier/Glorot initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        :param x: Input tokens (batch, seq_len)
        :return: Output predictions (batch, seq_len, vocab_size)
        """
        # Scaling embeddings to prevent them from being too small relative to positional encodings
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.layer_norm(x)
        x = self.output(x)
        return x

    def generate(self, start_tokens, max_length, temperature=1.0):
        """
        Autoregressive text generation

        :param start_tokens:(batch_size, seq_len) with starting tokens
        :param max_length: Max tokens to generate
        :param temperature: Sampling temperature
        :return: (batch_size, seq_len + max_len) generated tokens
        """
        self.eval()
        device = next(self.parameters()).device

        tokens = start_tokens.to(device)

        with torch.no_grad():
            for _ in range(max_length):
                # Use only last context_size tokens if too long
                input_tokens = tokens[:, -self.context_size:]
                logits = self(input_tokens)

                # Logits for last position
                logits = logits[:, -1, :] / temperature

                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to the seq
                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
