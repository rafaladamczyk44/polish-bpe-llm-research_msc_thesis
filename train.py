import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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


def get_config(config_name):
    """Dynamically get config class by name"""
    config_classes = {
        'Config18M': Config18M,
        'Config70M': Config70M,
        'Config85M': Config85M
    }

    if config_name not in config_classes:
        available = ', '.join(config_classes.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")

    return config_classes[config_name]


def parse_args():
    parser = argparse.ArgumentParser(description='Train Polish Language Model')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers (default: 0)')
    parser.add_argument('--max_epochs', type=int, default=3, help='Maximum number of epochs (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')

    # Model configuration
    parser.add_argument('--config', type=str, default='Config18M', choices=['Config18M', 'Config70M', 'Config85M'], help='Model configuration (default: Config18M)')

    # Data paths
    parser.add_argument('--data_path', type=str, default='training/data/tokenized_polish_wikipedia.pkl', help='Path to tokenized data')

    # Logging
    parser.add_argument('--project_name', type=str, default='master_thesis', help='Wandb project name (default: master_thesis)')

    # Checkpointing
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='training/models/checkpoints/', help='Directory to save checkpoints')

    # Hardware
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', '32', 'bf16-mixed'], help='Training precision (default: 16-mixed)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Get configuration
    CONFIG = get_config(args.config)
    print(f"Using configuration: {args.config}")
    print(f"Model size: {CONFIG.d_model}d, {CONFIG.num_layers} layers")

    # Setup dataset and dataloader
    dataset = WikiDataset(args.data_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Loaded dataset with {len(dataset)} sequences")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")

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
        callbacks=[checkpoint_callback],
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
    print("Starting training...")
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    print("Training completed!")


if __name__ == "__main__":
    main()
