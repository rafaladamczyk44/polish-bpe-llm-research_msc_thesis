from datasets import load_dataset
import re
import pickle

# Load
ds = load_dataset("chrisociepa/wikipedia-pl-20230401")
training_data =  ds['train']['text']

# Filter out short sentences
print('Filtering out short examples')
training_data = [text for text in training_data if len(text.strip()) > 50]

# Filter out non-Polish
print(f'Filtering out non Polish')

pol_pattern = r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s\.,!?;:()\[\]\-"\']'
filtered_training_data = []

for i, text in enumerate(training_data):
    # Remove non-polish patters
    filtered_text = re.sub(pol_pattern, ' ', text)

    # Normalize whitespace
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

    if text:  # Only keep non-empty texts
        filtered_training_data.append(filtered_text)


# Test
print(filtered_training_data[:5])

# Save to pickle
with open('training/data/filtered_polish_wikipedia.pkl', 'wb') as f:
    pickle.dump(filtered_training_data, f)

