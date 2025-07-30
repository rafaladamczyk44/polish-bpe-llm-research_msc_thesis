import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Create div_term for stable computation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

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