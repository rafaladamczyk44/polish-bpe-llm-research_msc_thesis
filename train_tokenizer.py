import pickle
from tokenizer_pl import BPETokenizerPL

VOCAB_SIZE = 30000
BATCH_SIZE = 5000
SAMPLE_SIZE = 1000000
MIN_FREQ = 2
TOKENIZER_PATH = "training/tokenizer/vocab"

with open('training/data/filtered_polish_wikipedia.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Initialize the Polish and rain the Polish tokenizer
tokenizer_pl = BPETokenizerPL(vocab_size=VOCAB_SIZE)

# Start the training
tokenizer_pl.train(training_data, sample_size=SAMPLE_SIZE, min_word_freq=MIN_FREQ, batch_size=BATCH_SIZE)

# Save to disk
tokenizer_pl.save_vocab(TOKENIZER_PATH)
