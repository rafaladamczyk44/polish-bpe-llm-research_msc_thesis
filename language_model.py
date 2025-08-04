import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Create div_term for stable computation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Single transformer block with MHA and FFN"""

    def __init__(self, d_model, num_heads, context_size, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, context_size, dropout)

        self.feed_forward_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Pass through the multihead attention and later through feed forward network with layer normalization
        :param x:
        :return:
        """
        # Pass through multihead attention
        attn_out = self.mha(x)

        #  Add droput
        x = x + self.dropout(attn_out)

        # Normalize
        x = self.layer_norm1(x)

        # FFN with residual connection to prevent missing gradients
        ffn_out = self.feed_forward_net(x)

        x = x + self.dropout(ffn_out)

        # Normalize
        x = self.layer_norm2(x)

        return x

class LanguageModel(nn.Module):
    """
    Language model which will test the thesis
    """
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 context_size,
                 d_ff,
                 dropout):
        super(LanguageModel, self).__init__()

        self.d_model = d_model
        self.context_size = context_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, context_size)

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
