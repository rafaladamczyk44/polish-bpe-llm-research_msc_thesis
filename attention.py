import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Masked self attention module for Transformer model
    """
    def __init__(self, d_in, d_out, context_size, dropout=0.1):
        """
        Initialize
        :param d_in: Input feature dimension
        :param d_out: Output feature dimension
        :param context_size: Sequence length
        :param dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)

        # using nn.Linear instead of nn.Parameter as it has optimized wight initialization
        self.Q = nn.Linear(d_in, d_out)
        self.K = nn.Linear(d_in, d_out)
        self.V = nn.Linear(d_in, d_out)

        # Registering the buffer for automatic move to the correct device
        self.register_buffer('mask', torch.triu(torch.ones(context_size, context_size, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        batch, seq_len, d_in = x.shape
        keys = self.K(x)
        queries = self.Q(x)
        values = self.V(x)

        # Attention(Q,K,V) = softmax( Q @ KT/ √dk) V
        attention_scores = queries @ keys.transpose(1, 2)

        # Adding masking for decoder
        attention_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], float('-inf'))

        # softmax( Q @ KT/ √dk)
        attention_weights = torch.softmax(attention_scores / (self.d_out ** 0.5), dim=-1)

        # Dropout
        attention_weights = self.dropout(attention_weights)

        # Multiply by values vector
        context_vector = attention_weights @ values

        return context_vector


class MultiHeadAttention(nn.Module):
    """
    Multi Head attention module for Transformer model
    """
    def __init__(self, d_model, num_heads, context_size, dropout=0.1):
        """
        Initialize the mha
        :param d_model: Model embedding dimension
        :param num_heads: Number of heads
        :param context_size: Sequence length
        :param dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        # Standard transformer constraint
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # Each head gets subset of dimensions

        self.heads = nn.ModuleList(
            [SelfAttention(d_model, self.d_head, context_size, dropout)
             for _ in range(num_heads)]
        )

        # Project to lin space for each head
        self.linear_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Calculate the outputs
        head_outputs = [head(x) for head in self.heads]

        # Concatenate the attention heads along the feature dimension
        context_vector = torch.cat(head_outputs, dim=-1)

        # Apply the projection back to the original dimension
        return self.dropout(self.linear_projection(context_vector))