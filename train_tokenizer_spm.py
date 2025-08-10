import pickle
import sentencepiece as spm
from config import VOCAB_SIZE

# https://github.com/google/sentencepiece

with open('training/data/filtered_polish_wikipedia.pkl', 'rb') as f:
    training_data = pickle.load(f)

print("Writing text file for SentencePiece...")
with open('training/data/text_for_spm.txt', 'w', encoding='utf-8') as f:
    for i, text in enumerate(training_data):
        f.write(text + '\n')

spm.SentencePieceTrainer.train(
    input='training/data/text_for_spm.txt',
    model_prefix='training/tokenizer/baseline_tokenizer',
    vocab_size=VOCAB_SIZE,
    model_type='bpe',
    shuffle_input_sentence=True,
    character_coverage=1.0,
    split_by_whitespace=True,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<bos>',
    eos_piece='<eos>',
    user_defined_symbols=['<mask>']
)