import re
import unicodedata
from collections import Counter
import json
import pickle
from pathlib import Path

"""
With a full Polish corpus, you'd expect:

Fertility rate: ~1.3-1.5 tokens/word (currently probably >2.0)
Common words: single tokens ("Polski", "jest", "nie")
Morphology: preserved splits ("na|pisać", "kot|ami")

"""

class BPETokenizerPL:
    """
    Polish tokenizer class.
    Preserves the morphology during BPE training, preventing merges across case endings, verb aspects,
    and other morphological features.
    """
    def __init__(self, vocab_size=30000):
        """
        Initialize the tokenizere
        :param vocab_size: Vocabulary size, default is 30k, inspired by Bielik3
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

        # Polish-specific characters
        self.pol_chars = set('ąćęłńóśżźĄĆĘŁŃÓŚŻŹ')
        self.pol_vowels = set('aąeęioóuy')

        # Special tokens for sequence tagging
        self.special_tokens = {
            '<pad>': 0,  # for sequence padding
            '<unk>': 1,  # unknown
            '<bos>': 2,  # beginning
            '<eos>': 3,  # end
            '<mask>': 4  # for masked attention
        }

        # Most common case endings
        # Not adding the nominative as it's the base form
        self.case_endings = {
            'gen': ['a', 'u', 'y', 'i', 'ów', 'ek'],  # Genitive
            'dat': ['owi', 'u', 'e', 'i', 'om'],      # Dative
            'acc': ['a', 'ę', 'o', 'e'],              # Accusative
            'ins': ['em', 'iem', 'om', 'ami', 'mi'],  # Instrumental
            'loc': ['e', 'u', 'i', 'ach', 'ech'],     # Locative
            'voc': ['e', 'u', 'o', 'i']               # Vocative

        }

        self.prefixes = [
            # Perfective prefixes (create completed actions)
            'na', 'za', 'po', 'wy', 'prze', 'przy', 'od', 'do', 's', 'z', 'u', 'roz', 'w', 'we', 'ob', 'obe', 'nad',
            'pod', 'przed', 'up',

            # Spatial/directional prefixes (additional ones not in perfective)
            'śród', 'między', 'ponad', 'poza', 'przez',

            # Negative/privative prefixes
            'nie', 'bez', 'anty', 'a', 'an',

            # Diminutive/attenuative prefixes (additional ones)
            'lekko',

            # Foreign/borrowed prefixes
            'arcy', 'auto', 'bio', 'de', 'dis', 'eko', 'euro', 'geo', 'hiper', 'in', 'inter', 'kontr', 'makro', 'meta',
            'mikro', 'multi', 'neo', 'para', 'proto', 'pseudo', 're', 'semi', 'sub', 'super', 'trans', 'ultra'
        ]

        # Verb endings for aspect detection
        self.verb_endings = {
            # Infinitive
            'infinitive': ['ać', 'eć', 'ić', 'yć', 'ować', 'ywać', 'iwać', 'nąć', 'ąć', 'ść', 'źć', 'c'],

            # Past tense (l-participle)
            'past': ['ał', 'ała', 'ało', 'ali', 'ały', 'ił', 'iła', 'iło', 'ili', 'iły', 'ął', 'ęła', 'ęło', 'ęli',
                     'ęły', 'łem', 'łeś', 'ła', 'ł', 'li', 'liście', 'łam', 'łaś', 'ło', 'łyśmy', 'łyście'],

            # Present tense (1st person singular)
            'present_1sg': ['ę', 'em', 'am', 'ym', 'im'],

            # Present tense (2nd person singular)
            'present_2sg': ['esz', 'isz', 'ysz', 'asz', 'sz'],

            # Present tense (3rd person singular)
            'present_3sg': ['e', 'i', 'y', 'a', 'je', 'ie'],

            # Present tense (1st person plural)
            'present_1pl': ['emy', 'imy', 'ymy', 'amy'],

            # Present tense (2nd person plural)
            'present_2pl': ['ecie', 'icie', 'ycie', 'acie'],

            # Present tense (3rd person plural)
            'present_3pl': ['ą', 'ją'],

            # Conditional
            'conditional': ['bym', 'byś', 'by', 'byśmy', 'byście', 'by'],

            # Imperative
            'imperative': ['', 'j', 'ij', 'yj', 'aj', 'my', 'cie', 'cie', 'jmy', 'jcie'],

            # Participles
            'active_participle': ['ący', 'ąca', 'ące', 'ący', 'ące'],
            'passive_participle': ['any', 'ana', 'ane', 'oni', 'one', 'ty', 'ta', 'te', 'ci', 'te'],
            'adverbial_participle': ['ąc', 'szy', 'łszy', 'wszy'],

            # Gerund (verbal noun)
            'gerund': ['anie', 'enie', 'cie', 'ęcie'],

            # Aspectual suffixes
            'imperfective_suffixes': ['ywać', 'iwać', 'ować', 'awać', 'ewać'],
            'perfective_suffixes': ['nąć', 'ąć', 'snąć'],

            # Archaic/formal endings
            'archaic_present': ['em', 'esz', 'e', 'emy', 'ecie', 'esz'],
            'archaic_past': ['ch', 'ech', 'och'],

            # Verbal adverbs
            'adverbial_present': ['ąc'],
            'adverbial_past': ['szy', 'wszy', 'łszy'],

            # Frequentative endings
            'frequentative': ['ywać', 'iwać', 'awać'],

            # Diminutive verb endings
            'diminutive': ['ić', 'yć', 'ować'],
        }

        # Morphological alternations
        self.alternations = {
            # Vowel
            'ó_o': [('ó', 'o')],  # ex. wóz -> wozy, król -> królowie
            'ą_ę': [('ą', 'ę')],  # ex. ręka -> rąk, wąs -> wąsy
            'e_a': [('e', 'a')],  # ex. siedem -> siódmy
            'o_a': [('o', 'a')],  # ex. kot -> kota
            'y_i': [('y', 'i')],  # ex. dobry -> dobrzy

            # Palatization
            'palatalization_k_c': [('k', 'c')],  # ex. ręka -> ręce
            'palatalization_g_dz': [('g', 'dz')],  # ex. noga -> nodze
            'palatalization_ch_sz': [('ch', 'sz')],  # ex. mucha -> musze
            'palatalization_t_ci': [('t', 'ci')],  # ex. kret -> krecie
            'palatalization_d_dzi': [('d', 'dzi')],  # ex. sąd -> sądzie
            'palatalization_ń_ni': [('ń', 'ni')],  # ex. dzień -> dnia
            'palatalization_ł_l': [('ł', 'l')],  # ex. stół -> stole
            'palatalization_s_si': [('s', 'si')],  # ex. las -> lesie
            'palatalization_z_zi': [('z', 'zi')],  # ex. mróz -> mrozie
            'b_bi': [('b', 'bi')],  # ex. grób -> grobie
            'p_pi': [('p', 'pi')],  # ex. sklep -> sklepie
            'w_wi': [('w', 'wi')],  # ex. lew -> lwie
            'f_fi': [('f', 'fi')],  # ex. szef -> szefie
            'm_mi': [('m', 'mi')],  # ex. dom -> domie

            # Clusters
            'st_ść': [('st', 'ść')],  # ex. most -> moście
            'zd_ździ': [('zd', 'ździ')],  # ex. gwiazda -> gwieździe
            'sł_śl': [('sł', 'śl')],  # ex. masło -> maśle
            'zł_źl': [('zł', 'źl')],  # ex. kozła -> koźle

            # Iotation
            'iotation_t_ci': [('t', 'ci')],  # ex. liść -> liście
            'iotation_d_dzi': [('d', 'dzi')],  # ex. sąsiad -> sąsiedzi
            'iotation_s_si': [('s', 'si')],  # ex. gość -> goście
            'iotation_z_zi': [('z', 'zi')],  # ex. książę -> książęta

        }

        # Add protected words that should never be split
        self.protected_words = [
            # Conjunctions
            'i', 'oraz', 'a', 'ale', 'lecz', 'lub', 'albo', 'czy', 'ani', 'ni',
            'że', 'żeby', 'aby', 'by', 'gdyby', 'jeśli', 'jeżeli', 'gdy', 'kiedy',
            'ponieważ', 'bo', 'bowiem', 'albowiem', 'gdyż', 'więc', 'zatem',
            'jednak', 'natomiast', 'choć', 'chociaż', 'mimo', 'pomimo',

            # Prepositions
            'w', 'na', 'do', 'z', 'o', 'po', 'za', 'przed', 'nad', 'pod', 'przy',
            'bez', 'dla', 'od', 'u', 'ze', 'przez', 'między', 'wobec',

            # Particles & pronouns
            'się', 'nie', 'już', 'jeszcze', 'też', 'także', 'tylko', 'nawet',
            'to', 'ten', 'ta', 'te', 'tym', 'tego', 'tej', 'tych',
            'ja', 'ty', 'on', 'ona', 'ono', 'my', 'wy', 'oni', 'one',

            # Question words
            'co', 'kto', 'gdzie', 'kiedy', 'dlaczego', 'jak', 'który', 'która', 'które',

            # Common adverbs
            'bardzo', 'może', 'tutaj', 'tam', 'teraz', 'wtedy', 'zawsze', 'nigdy'
        ]

        self.next_token_id = len(self.special_tokens)

    # Pre-processing
    def normalize(self, text: str):
        """
        Normalize Polish text, keep the diacritics

        :param text: Input text
        :return: Normalized text
        """

        # Keeping the polish chars
        text = unicodedata.normalize('NFC', text)
        text = text.lower()

        # Standardize the whitespaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def preprocess(self, text):
        """
        Basic word tokenization of normalized text
        Split on whitespace and punctuation, but keep punctuation as separate tokens

        :param text: Input text
        :return: List of pre-processed tokens
        """

        text = self.normalize(text)

        pattern = r'(\w+|[^\w\s])'
        tokens = re.findall(pattern, text, re.UNICODE)
        return tokens

    # Training
    def _identify_morpheme_boundaries(self, word: str):
        """
        This method analyzes a Polish word to find positions where morphemes (meaningful units)
        are joined together. It identifies:
        - Boundaries between stems and case endings (e.g., kot|a, dom|u)
        - Boundaries between verb prefixes and roots (e.g., na|pisać, wy|jść)
        - Boundaries between verb stems and inflectional endings (e.g., pis|ać, czyt|am)

        The boundaries are used during BPE training to prevent merging across morphological boundaries,
        preserving the linguistic structure of Polish words.

        How it works:
        1. Skip analysis for protected words (pronouns, prepositions, etc.)
        2. Check for verb prefixes at word beginning (na-, prze-, wy-, etc.)
        3. Scan for verb inflectional endings from word end (past, present, conditional, etc.)
        4. Scan for noun case endings from word end (genitive, dative, etc.)
        5. Validate stems contain Polish vowels before adding boundaries
        6. Return all valid boundary positions

        Example:
        "napisałem" → boundaries at positions [2, 5, 7] → "na|pis|ał|em"
        - Position 2: prefix boundary (na|pisałem)
        - Position 5: past tense boundary (napisal|em)
        - Position 7: person marker boundary (napisał|em)

        :param word: Polish word to analyze
        :return: Set of boundary positions (integers indicating character positions)
        """

        # Skip boundary detection for protected words
        if word.lower() in self.protected_words:
            return set()

        boundaries = set()
        word_lower = word.lower()

        # 1. VERB PREFIX DETECTION

        # Skip derivational categories that create new words rather than inflect existing ones
        inflectional_categories = [
            cat for cat in self.verb_endings.keys()
                if cat not in ['imperfective_suffixes', 'perfective_suffixes']
            ]

        # Check for verb prefixes at the beginning of the word
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            # Make sure it's the preffix not ending
            if word_lower.startswith(prefix) and len(word) > len(prefix) + 3:

                # Rest of the word
                remaining = word_lower[len(prefix):]

                # Check if what follows could be a verb stem
                if any(remaining.endswith(ending) for category in inflectional_categories
                       for ending in self.verb_endings[category] if ending):

                    # Add to boundary if yes
                    boundaries.add(len(prefix))

                    break  # Take the longest matching prefix

        # 2. VERB ENDING DETECTION
        found_endings = []

        # Find all possible endings and calculate possible boundaries
        for category in inflectional_categories:
            for ending in sorted(self.verb_endings[category], key=len, reverse=True):
                # If found ending
                if ending and word_lower.endswith(ending):
                    # Move the boundary
                    boundary_pos = len(word) - len(ending)

                    # Add possible boundaries
                    if boundary_pos > 0:
                        found_endings.append((boundary_pos, ending, category))

        # Process verb endings (longest first)
        if found_endings:
            found_endings.sort(key=lambda x: x[0], reverse=True)

            for pos, ending, category in found_endings:
                if pos > 1:  # Ensure minimum viable stem
                    stem = word_lower[:pos]
                    # Validate stem: should contain Polish vowels
                    if any(char in self.pol_vowels for char in stem):
                        boundaries.add(pos)

        # 3. NOUN CASE ENDING DETECTION
        # Process all case endings (can coexist with verb boundaries for ambiguous words)
        for case, endings in self.case_endings.items():
            for ending in sorted(endings, key=len, reverse=True):
                if word_lower.endswith(ending) and len(word) > len(ending):
                    boundary_pos = len(word) - len(ending)

                    if boundary_pos > 1:  # Minimum stem length
                        stem = word_lower[:boundary_pos]

                        # Validate stem: should contain Polish vowels
                        if any(char in self.pol_vowels for char in stem):
                            boundaries.add(boundary_pos)
                            break  # Take longest valid ending per case

        return boundaries

    def _word_frequency(self, corpus):
        """
        Calculate the frequency of each word in a corpus.

        :param corpus: Input dataset
        :return: Word frequencies
        """
        word_freq = Counter()

        for text in corpus:
            tokens = self.preprocess(text)
            word_freq.update(tokens)

        return word_freq

    def _get_pairs(self, word_tokens):
        """
        Get all adjacent pairs from tokenized word.

        :param word_tokens: List of tokens representing a word

        :returns: List of adjacent token pairs
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def _merge_pair(self, pair, word_tokens):
        """
        Merge pairs in all word tokens.

        :param pair: Pair of tokens to merge
        :param word_tokens: List of tokenized words

        :return:Updated word tokens with pair merged
        """

        new_word_tokens = []

        for tokens in word_tokens:
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_word_tokens.append(new_tokens)

        return new_word_tokens

    def train(self, corpus, verbose=True):
        """
        Train BPE on corpus with morphological constraints.

        :param corpus: List of text documents
        :param verbose: Whether to print training progress
        """
        # Get word frequencies
        word_freq = self._word_frequency(corpus)

        # Initialize vocabulary with characters and special tokens
        self.vocab = self.special_tokens.copy()

        # Add all characters to vocabulary
        chars = set()
        for word in word_freq:
            chars.update(word)

        for char in sorted(chars):
            # Move to next token
            self.vocab[char] = self.next_token_id
            self.next_token_id += 1

        # Prepare words for BPE (character-level tokenization)
        word_tokens = []
        word_boundaries = []
        word_counts = []

        # Add protected words directly to vocabulary first
        for word in self.protected_words:
            if word not in self.vocab:
                self.vocab[word] = self.next_token_id
                self.next_token_id += 1

        for word, count in word_freq.items():

            # Skip if protected word
            if word in self.protected_words:
                continue

            # Skip punctuation and very short words
            if len(word) > 1 and any(c.isalpha() for c in word):
                tokens = list(word)
                boundaries = self._identify_morpheme_boundaries(word)

                word_tokens.append(tokens)
                word_boundaries.append(boundaries)
                word_counts.append(count)

        if verbose:
            print(f"Training on {len(word_tokens)} unique words")
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # BPE training loop
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Count all pairs
            pair_counts = Counter()
            for tokens, boundaries, count in zip(word_tokens, word_boundaries, word_counts):
                pairs = self._get_pairs(tokens)

                for pair in pairs:
                    # Find ALL occurrences of this pair in the word
                    for i in range(len(tokens) - 1):
                        if (tokens[i], tokens[i + 1]) == pair:
                            # Calculate position
                            pos = sum(len(tokens[j]) for j in range(i))
                            # Check if merge would cross morpheme boundary
                            if pos + len(tokens[i]) not in boundaries:
                                pair_counts[pair] += count

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Merge the pair
            word_tokens = self._merge_pair(best_pair, word_tokens)

            # Add to vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = self.next_token_id
                self.next_token_id += 1
                self.merges.append(best_pair)

            iteration += 1
            print(f'Running iteration {iteration}')
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Vocab size = {len(self.vocab)}, Merged pairs = {len(pair_counts)}")
                print(f"Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'")

        if verbose:
            print(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    # Main used methods
    def encode(self, text: str):
        """
        Encode text to token IDs.

        :param text: Text to encode

        :returns:List of token IDs
        """
        tokens = self.preprocess(text)
        encoded = []

        for token in tokens:
            # Try to find token in vocabulary
            if token in self.vocab:
                encoded.append(self.vocab[token])
                continue

            # Apply BPE merges
            chars = list(token)
            for merge in self.merges:
                i = 0
                while i < len(chars) - 1:
                    if (chars[i], chars[i + 1]) == merge:
                        chars[i] = chars[i] + chars[i + 1]
                        chars.pop(i + 1)
                    else:
                        i += 1

            # Encode resulting subwords
            for subword in chars:
                if subword in self.vocab:
                    encoded.append(self.vocab[subword])
                else:
                    # Handle unknown subwords character by character
                    for char in subword:
                        if char in self.vocab:
                            encoded.append(self.vocab[char])
                        else:
                            encoded.append(self.vocab['<unk>'])

        return encoded

    def decode(self, token_ids):
        """
        Decode token IDs back to text.

        :param token_ids: List of token IDs

        :returns: Decoded text
        """
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}

        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                # Skip special tokens in decoding
                if token not in self.special_tokens:
                    tokens.append(token)

        # Join tokens with appropriate spacing
        text = ''
        for i, token in enumerate(tokens):
            if i > 0 and token.isalnum() and tokens[i - 1].isalnum():
                text += ' '
            text += token

        return text

    def tokenize(self , text: str):
        """
        Tokenize text and return the actual subword strings (not IDs).
        :param text: Text to be tokenized
        :return: List of tokenized sub-words
        """

        encoded = self.encode(text)

        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token[id] for id in encoded if id in id_to_token]

        return tokens

    # Metrics
    def calculate_fertility(self, test_corpus):
        """
        How many tokens per word on average?
        """
        total_words = 0
        total_tokens = 0

        for text in test_corpus:
            words = text.split()
            tokens = self.encode(text)
            total_words += len(words)
            total_tokens += len(tokens)

        fertility = total_tokens / total_words
        print(f"Fertility rate: {fertility:.2f} tokens/word")
        # Good range: 1.2-1.8 for morphologically rich languages
        return fertility

    def calculate_coverage(self, test_corpus):
        """
        What percentage of text can we encode without <unk> tokens?
        """
        total_tokens = 0
        unknown_tokens = 0

        for text in test_corpus:
            encoded = self.encode(text)
            total_tokens += len(encoded)
            unknown_tokens += encoded.count(self.vocab.get('<unk>', 1))

        coverage = 1 - (unknown_tokens / total_tokens)
        print(f"Coverage: {coverage:.1%}")
        # Should be > 99% on in-domain text
        return coverage

    # Utils
    def save_vocab(self, path: str, format: str = 'json'):
        """
        Save the trained tokenizer state to disk.

        :param path: Base path for saving (without extension)
        :param format: Save format ('json', 'pickle', or 'both')
        """
        path = Path(path)

        # Prepare tokenizer state
        tokenizer_state = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges,
            'next_token_id': self.next_token_id,
            'special_tokens': self.special_tokens,
            # Save linguistic data for reproducibility
            'pol_chars': list(self.pol_chars),
            'case_endings': self.case_endings,
            'protected_words': list(self.protected_words),
            'verb_endings': self.verb_endings,
            'alternations': self.alternations
        }

        if format in ['json', 'both']:
            # Save as JSON (human-readable, cross-platform)
            json_path = path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)
            print(f"Saved tokenizer to {json_path}")

        if format in ['pickle', 'both']:
            # Save as pickle (preserves Python objects exactly)
            pickle_path = path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(tokenizer_state, f)
            print(f"Saved tokenizer to {pickle_path}")

        # Save vocab only (for HuggingFace compatibility)
        vocab_path = path.with_suffix('.vocab')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, token_id in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{token_id}\n")
        print(f"Saved vocabulary to {vocab_path}")

    def load_vocab(self, path: str, format: str = 'json'):
        """
        Load a previously saved tokenizer state.

        :param path: Path to the saved tokenizer file
        :param format: Load format ('json' or 'pickle')
        """
        path = Path(path)

        if format == 'json':
            if not path.suffix:
                path = path.with_suffix('.json')
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        elif format == 'pickle':
            if not path.suffix:
                path = path.with_suffix('.pkl')
            with open(path, 'rb') as f:
                state = pickle.load(f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")

        # Restore tokenizer state
        self.vocab_size = state['vocab_size']
        self.vocab = state['vocab']
        self.merges = state['merges']
        self.next_token_id = state['next_token_id']
        self.special_tokens = state['special_tokens']

        # Restore linguistic data
        self.pol_chars = set(state['pol_chars'])
        self.case_endings = state['case_endings']
        self.protected_words = set(state['protected_words'])
        self.verb_endings = state['verb_endings']
        self.alternations = state['alternations']

        print(f"Loaded tokenizer from {path}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    @classmethod
    def from_pretrained(cls, path: str, format: str = 'json'):
        """
        Class method to create a tokenizer instance from saved state.

        :param path: Path to saved tokenizer
        :param format: Load format ('json' or 'pickle')
        :return: Loaded tokenizer instance
        """
        tokenizer = cls()
        tokenizer.load_vocab(path, format)
        return tokenizer

