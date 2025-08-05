# Impact of Polish Specific Tokenization on LLM Performance

This repository contains the implementation for Master Thesis research investigating how morphologically aware tokenization affects the accuracy and performance of Language Models when working with Polish data.

## Abstract

Tokenization is an inseparable part of Large Language Model architecture. However, as one can observe, the majority of tokenizers specialize in English, with a little attention for under-represented languages' specificity. This research aims to develop the BPE tokenizer trained and adjusted specifically for Polish language, and measure the impact of language-specific tokenization on the performance of Language Model, when trained on the Polish corpora dataset. The tokenizer will be sensitive to Polish morphology to allow the model to accurately distinguish between the person, number and the case of each word. Improved convergence could enable a possibility of training large scale Language models with less resources for a given language. This work contributes to democratizing LLM development for morphologically rich languages by reducing computational requirements.

The project consists of:
- Polish BPE tokenizer with morphological awareness
- Transformer based LLM architecture
- Training and data processing pipelines

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Download Training Data
```bash
pip install gdown && gdown [GOOGLE_FILE_ID] -O training/data/tokenized_polish_wikipedia.pkl
```

## Data Preparation

If you want to prepare your own data from scratch:

1. **Process raw Wikipedia data:**
```bash
python process_data.py
```

2. **Train custom tokenizer:**
```bash
python train_tokenizer.py
```

3. **Tokenize the processed data:**
```bash
python process_data_tokenization.py
```

## Training Arguments

**Training Hyperparameters:**
- `--batch_size` (int, default: 4): Batch size for training
- `--num_workers` (int, default: 0): Number of dataloader workers
- `--max_epochs` (int, default: 3): Maximum number of epochs
- `--learning_rate` (float, default: 3e-4): Learning rate

**Model Configuration:**
- `--config` (str, default: Config18M): Model configuration
  - Choices: `Config18M`, `Config70M`, `Config85M`

**Data Paths:**
- `--data_path` (str, default: training/data/tokenized_polish_wikipedia.pkl): Path to tokenized data

**Logging:**
- `--project_name` (str, default: master_thesis): Wandb project name

**Checkpointing:**
- `--resume_from` (str, default: None): Path to checkpoint to resume from
- `--checkpoint_dir` (str, default: training/models/checkpoints/): Directory to save checkpoints

**Hardware:**
- `--devices` (int, default: 1): Number of GPUs to use
- `--precision` (str, default: 16-mixed): Training precision
  - Choices: `16-mixed`, `32`, `bf16-mixed`

## Training

### Basic Training
```bash
python train.py
```

### RTX 3090 (18M Parameters)
```bash
python train.py --config Config18M --batch_size 64 --devices 1 --precision bf16-mixed --num_workers 4
```

### RTX 4090 (70M Parameters)
```bash
python train.py --config Config70M --learning_rate 2e-4 --batch_size 128 --devices 1 --precision bf16-mixed --num_workers 16
```

### Resume from Checkpoint
```bash
python train.py --resume_from training/models/checkpoints/model-epoch-05.ckpt
```

### Multi-GPU Training
```bash
python train.py --devices 2 --batch_size 128 --precision bf16-mixed
```