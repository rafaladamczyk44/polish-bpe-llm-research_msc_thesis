class Config18M:
    vocab_size = 30000
    d_model = 256
    num_heads = 8
    num_layers = 4
    context_size = 512
    d_ff = 4*d_model

class Config70M:
    vocab_size = 30000
    d_model = 512
    num_heads = 8
    num_layers = 6
    context_size = 512
    d_ff = 4*d_model

class Config85M:
    vocab_size = 30000
    d_model = 768
    num_heads = 12
    num_layers = 6
    context_size = 512
    d_ff = 4*d_model

