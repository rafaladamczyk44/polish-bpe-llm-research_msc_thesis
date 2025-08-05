## Imapct of Polish specific tokenization on LLM performance

This is the repository for my Master Thesis research on how does the morphologically aware tokenization affects the accuracy and performance of Language Models when working with Polish data

The project consists of:
- Polish BPE tokenizer
- Transformer based LLM architecture
- Training and data processing pipelines

### Installation/setup

1. Download the tokenized dataset  
pip install gdown && gdown [GOOGLE FILE ID] -O training/data/tokenized_polish_wikipedia.pkl
2. Run with the specified configuration:
   1. For 3090 GPU with 18M parameters:  
   ```bash 
   python train.py --config Config18M --batch_size 64 --devices 1 --precision bf16-mixed --num_workers 4
   ```
   2. For 4090 GPU with 70M parameters:  
   ```bash 
   python train.py --config Config70M --learning_rate 2e-4 --batch_size 128 --devices 1 --precision bf16-mixed --num_workers 16
   ```

### Training Arguments

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

#### Usage Examples

**Basic training:**
```bash
python train.py