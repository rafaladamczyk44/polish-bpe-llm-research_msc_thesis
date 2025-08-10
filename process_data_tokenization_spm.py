import pickle
import sentencepiece as spm

# Load the tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('training/tokenizer/baseline_tokenizer.model')

# Read the text file and tokenize
tokenized_data = []

with open('training/data/text_for_spm.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # Read all lines first

print(f"Total lines to tokenize: {len(lines)}")

for i, line in enumerate(lines):
    if i % 1000 == 0:  # Progress tracking
        print(f'Tokenizing {i}/{len(lines)}...')

    # Strip newline and encode
    text = line.strip()
    if text:  # Skip empty lines
        tokenized_text = tokenizer.encode(text, out_type=int)
        tokenized_data.append(tokenized_text)

print(f"Tokenized {len(tokenized_data)} sentences")

# Save tokenized data
with open('training/data/tokenized_polish_wikipedia_spm.pkl', 'wb') as f:
    pickle.dump(tokenized_data, f)

print("Tokenization complete!")