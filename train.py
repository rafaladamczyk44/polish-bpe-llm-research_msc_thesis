import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger

run = wandb.init(
    project="master_thesis",
)

wandb_logger = WandbLogger()
trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
