import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from tokenizer_pl import BPETokenizerPL
from language_model import LanguageModel
from config import *

run = wandb.init(
    project="master_thesis",
)

wandb_logger = WandbLogger()
trainer = L.Trainer(max_epochs=10, logger=wandb_logger)

config = Config25M

tokenizer_pl = BPETokenizerPL(config.vocab_size)

model = LanguageModel(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    context_size=config.context_size,
    d_ff=config.d_ff,
    dropout=0.1,
)