from datasets import load_dataset
from tokenizer_pl import BPETokenizerPL
import random
import re


VOCAB_SIZE = 30000
TOKENIZER_PATH = "tokenizers_vocab/"

# Download the dataset
ds = load_dataset("chrisociepa/wikipedia-pl-20230401")

# Filter your dataset
training_data =  ds['train']['text']
training_data = [text for text in training_data if len(text.strip()) > 50]

# Initialize the Polish and rain the Polish tokenizer
tokenizer_pl = BPETokenizerPL(vocab_size=VOCAB_SIZE)
print('Starting the training')

tokenizer_pl.train(training_data, sample_size=1000000, verbose=True)

# Save to disk
tokenizer_pl.save_vocab(TOKENIZER_PATH)

"""
# Run evaluation
tokenizer.calculate_coverage(test_corpus)
tokenizer.calculate_fertility(test_corpus)
test_words = ["mógłbym", "Polski", "kotami", "napisać"]
for word in test_words:
    tokens = tokenizer.tokenize(word)
    print(f"{word} → {' | '.join(tokens)}")
"""


