import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from config import Config18M

config = Config18M()

class WikiDataset(Dataset):
    def __init__(self, path):
        self.data = self._load_pkl(path=path)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _load_pkl(self, path):
        """Load tokenized(!) data"""
        with open(path, 'rb') as f:
            training_data = pickle.load(f)

        return training_data

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to the same length within each batch
    """
    # Trimming the sequence length
    batch = [seq[:config.context_size] for seq in batch]
    return pad_sequence(batch, batch_first=True, padding_value=0)