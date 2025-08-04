from tokenizer_pl import BPETokenizerPL
import pickle

with open('training/data/filtered_polish_wikipedia.pkl', 'rb') as f:
    training_data = pickle.load(f)

tokenizer = BPETokenizerPL().from_pretrained('training/tokenizer/vocab.json')
tokenized_data = []

for i, text in enumerate(training_data):
    if i % 1000 == 0:  # Progress tracking
        print(f'Tokenizing {i}/{len(training_data)}...')

    tokenized_text = tokenizer.encode(text)
    tokenized_data.append(tokenized_text)

with open('training/data/tokenized_polish_wikipedia.pkl', 'wb') as f:
    pickle.dump(tokenized_data, f)