import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from tokenizer_pl import BPETokenizerPL
from language_model import LanguageModel
from config import *
from utils import WikiDataset, collate_fn


class LMTraining(L.LightningModule):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 context_size,
                 d_ff,
                 dropout,
                 learning_rate=3e-4
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            context_size=context_size,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Up to the last token
        inputs = batch[:, :-1]
        # Shift to all but first token
        targets = batch[:, 1:]

        # Forward pass
        logits = self(inputs)  # [batch_size, seq_len-1, vocab_size]

        loss = F.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),  # [batch*seq, vocab_size]
            target=targets.reshape(-1),  # [batch*seq]
            ignore_index = 0 # ignore padding
        )

        perplexity = torch.exp(loss)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

BATCH_SIZE = 4
NUM_WORKERS = 0
CONFIG = Config25M

dataset = WikiDataset('training/data/tokenized_polish_wikipedia.pkl')
train_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn
)


model = LMTraining(
    vocab_size=CONFIG.vocab_size,
    d_model=CONFIG.d_model,
    num_heads=CONFIG.num_heads,
    num_layers=CONFIG.num_layers,
    context_size=CONFIG.context_size,
    d_ff=CONFIG.d_ff,
    dropout=0.1,
)

trainer = L.Trainer(
    max_epochs=3,
    logger=WandbLogger(project="master_thesis"),
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed",  # Automatic mixed precision
    gradient_clip_val=1.0,  # Gradient clipping
    # val_check_interval=1000,  # Validate every 1000 steps
)

trainer.fit(model, train_dataloader)
