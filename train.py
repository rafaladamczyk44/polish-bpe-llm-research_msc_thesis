import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, LearningRateMonitor

from language_model import LanguageModel
from utils import *


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

    def validation_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = self(inputs)

        loss = F.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),
            target=targets.reshape(-1),
            ignore_index=0
        )

        perplexity = torch.exp(loss)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)

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


def main():
    args = parse_args()
    CONFIG = get_config(args.config)

    # Setup dataset and dataloader
    dataset = WikiDataset(args.data_path)

    # Split data 4:1
    train_data, val_data = train_test_split(dataset, train_size=0.8)

    # Training
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Validation
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = LMTraining(
        vocab_size=CONFIG.vocab_size,
        d_model=CONFIG.d_model,
        num_heads=CONFIG.num_heads,
        num_layers=CONFIG.num_layers,
        context_size=CONFIG.context_size,
        d_ff=CONFIG.d_ff,
        dropout=0.1,
        learning_rate=args.learning_rate
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )

    # Setup logger
    logger = WandbLogger(project=args.project_name)

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy='ddp' if args.devices > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            RichProgressBar(),
            LearningRateMonitor(logging_interval='step')
        ],
        enable_checkpointing=True,
    )

    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume_from:
        ckpt_path = args.resume_from
        print(f"Resuming from checkpoint: {ckpt_path}")
    elif os.path.exists(os.path.join(args.checkpoint_dir, 'last.ckpt')):
        ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
        print(f"Resuming from last checkpoint: {ckpt_path}")

    # Start training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
