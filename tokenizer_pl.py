import gc
import re
import json
import pickle
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict


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

    def _normalize_and_preprocess(self, text):
        """
        Normalize Polish text, keep the diacritics
        Split on whitespace and punctuation, but keep punctuation as separate tokens

        :param text: Input text
        :return: List of pre-processed tokens
        """
        # Keeping the polish chars
        text = unicodedata.normalize('NFC', text)
        text = text.lower()

        # Standardize the whitespaces
        text = re.sub(r'\s+', ' ', text)

        text = text.strip()

        pattern = r'(\w+|[^\w\s])'
        tokens = re.findall(pattern, text, re.UNICODE)
        return tokens

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

    def _word_frequency(self, corpus, sample_size, min_word_freq, batch_size):
        """
        Split corpus into batches, apply Polish filtering and count words in each batch
        :param corpus:
        :param sample_size:
        :param min_word_freq:
        :param batch_size:
        :return:
        """
        word_freq = Counter()
        processed_docs = 0

        # Process in batches
        for i in range(0, len(corpus), batch_size):
            # 0 to 10k batch
            batch = corpus[i:i+batch_size]
            for text in batch:
                tokens = self._normalize_and_preprocess(text)
                word_freq.update(tokens)

            processed_docs += len(batch)

            print(f'Processed {processed_docs} docs')

            # Garbage collector
            if processed_docs % (batch_size * 10) == 0:
                gc.collect()

        # Count word frequency
        word_freq = {
            word: count for word, count in word_freq.items() if count >= min_word_freq
        }

        # Order top words in sample size
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:sample_size]

        return dict(top_words)

    def _get_pairs(self, words):
        """
        Get all adjacent pairs from tokenized word.
        Update: using pre-computed positions for optimizing

        :param words: List of tokens representing a word
        :returns: List of adjacent token pairs
        """
        pair_counts = defaultdict(int)

        for tokens, boundaries, count in words:
            # Initial position
            positions = [0]

            for token in tokens:
                # Add the tokens
                positions.append(positions[-1] + len(token))

            for i in range(len(tokens) - 1):
                merge_position = positions[i + 1]
                if merge_position not in boundaries:
                    pair = (tokens[i], tokens[i+1])
                    pair_counts[pair] += count

        return pair_counts

    def _merge_pair(self, pair, word_data):
        """
        Merge pairs in all word tokens.
        Update: Merge in place to save memory

        :param pair: Pair of tokens to merge
        :param word_data: List of tokenized words

        :return:Updated word tokens with pair merged
        """

        target_first, target_second = pair
        new_token = target_first + target_second

        for i, (tokens, boundaries, count) in enumerate(word_data):
            new_tokens = []
            j = 0

            while j < len(tokens):

                if j < len(tokens) - 1 and tokens[j] == target_first and tokens[j+1] == target_second:
                    new_tokens.append(new_token)
                    # Jump to next pair pos
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1

            word_data[i] = (new_tokens, boundaries, count)

    def train(self, corpus, sample_size, min_word_freq, batch_size, save_interval=500, verbose=True):
        """
        Train BPE on corpus with morphological constraints and optional sampling.

        This method implements Byte-Pair Encoding (BPE) with Polish-specific morphological awareness.
        Unlike standard BPE, it prevents merging across morpheme boundaries, ensuring that:
        - Stems remain intact (e.g., "kot" in "kota", "kotów")
        - Prefixes stay separate from roots (e.g., "na|pisać", "prze|czytać")
        - Inflectional endings are preserved (e.g., "czyt|am", "dom|u")

        Training process:
        1. Filter corpus to retain only Polish text using regex pattern matching
        2. Calculate word frequencies and apply sampling/filtering thresholds
        3. Initialize vocabulary with special tokens (<pad>, <unk>, etc.) and all characters
        4. For each word, identify morpheme boundaries using linguistic rules
        5. Iteratively find most frequent adjacent token pairs that don't cross boundaries
        6. Merge the best pair across all words and add to vocabulary
        7. Repeat until target vocabulary size is reached

        Morphological constraints prevent linguistically invalid merges:
        - "kot|a" won't become "kota" (preserves stem-ending boundary)
        - "na|pisać" won't become "napisać" (preserves prefix-root boundary)
        - "czyt|am" won't become "czytam" (preserves stem-inflection boundary)

        Memory optimization techniques:
        - Pre-compute character positions once per word to avoid recalculation
        - Use frequency weighting to prioritize common patterns
        - Optional sampling to limit training set size for large corpora

        Example training progression:
        Initial: ['k', 'o', 't', 'a'] with boundary at position 3
        Iteration 1: ('k', 'o') → ['ko', 't', 'a']
        Iteration 2: ('k', 'o', 't') → ['kot', 'a'] (stops at boundary)
        Final result: "kot|a" structure preserved

        :param corpus: List of text documents
        :param verbose: Whether to print training progress
        :param sample_size: Maximum number of unique words to use for training (None = use all)
        :param min_word_freq: Minimum frequency threshold for words (filters rare words)
        :param batch_size: Number of words to train on at a time
        :param save_interval: Number of words to save at a time
        :param checkpoint_path: Path to checkpoint file
        """

        # Get word frequencies
        if verbose:
            print("Calculating word frequencies...")

        word_freq = self._word_frequency(corpus, sample_size, min_word_freq, batch_size)
        # Force garbage collection after corpus processing
        gc.collect()

        if verbose:
            print(f"Word frequency calculation complete. Using {len(word_freq):,} unique words")


        # Initialize vocabulary with characters and special tokens
        self.vocab = self.special_tokens.copy()

        # Add protected words directly to vocabulary first
        for word in self.protected_words:
            if word not in self.vocab:
                self.vocab[word] = self.next_token_id
                self.next_token_id += 1

        # Add all characters to vocabulary
        chars = set()
        for word in word_freq:
            chars.update(word)

        # Get to the first open position
        for char in sorted(chars):
            self.vocab[char] = self.next_token_id
            self.next_token_id += 1

        # Prepare words for BPE (character-level tokenization)
        word_data = [] # for char level

        for word, count in word_freq.items():
            # Skip if protected word
            if word in self.protected_words:
                continue

            # Skip words shorter than 1 char and punctuation/numbers
            if len(word) > 1 and any(c.isalpha() for c in word):
                tokens = list(word)

                # For other words,
                # Apply the boundary logic
                boundaries = self._identify_morpheme_boundaries(word)

                # Add tokens and boundaries
                word_data.append((tokens, boundaries, count))


        # Free up the memory
        del word_freq
        gc.collect()

        if verbose:
            print(f"Training on {len(word_data)} unique words")
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # BPE training loop
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Count all pairs in single pass
            pair_counts = self._get_pairs(word_data)

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Merge the pair for all vocab
            self._merge_pair(best_pair, word_data)

            # Add to vocabulary and move to the next position in vocab
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = self.next_token_id
                self.next_token_id += 1
                self.merges.append(best_pair)

            iteration += 1

            print(f'Running iteration {iteration}, vocabulary size: {len(self.vocab)}')
            if iteration % save_interval == 0:
                self._save_checkpoint(f"training/tokenizer/checkpoints/checkpoint_{iteration}.pkl", iteration)

            if iteration % save_interval == 0:
                gc.collect()

            if verbose and iteration % save_interval == 0:
                print(f"Iteration {iteration}: Vocab size = {len(self.vocab)}, Merged pairs = {len(pair_counts)}")
                print(f"Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'")

        if verbose:
            print(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    def resume_training_from_corpus(self, checkpoint_path, corpus, sample_size, min_word_freq, batch_size, save_interval=500, verbose=True):
        """
        Resume training from existing checkpoint by reconstructing word_data from corpus

        :param checkpoint_path: Path to existing checkpoint
        :param corpus: Original corpus (same as used for initial training)
        :param sample_size: Same parameters as initial training
        :param min_word_freq: Same parameters as initial training
        :param batch_size: Batch size for corpus processing
        :param verbose: Whether to print progress
        :param save_interval: Save checkpoint every N iterations
        """
        # Load checkpoint state
        iteration = self._load_checkpoint(checkpoint_path)

        if verbose:
            print("Reconstructing training data from corpus...")

        # Recreate word_data exactly as in initial training
        word_freq = self._word_frequency(corpus, sample_size, min_word_freq, batch_size)

        word_data = []  # for char level

        for word, count in word_freq.items():
            # Skip if protected word
            if word in self.protected_words:
                continue

            # Skip words shorter than 1 char and punctuation/numbers
            if len(word) > 1 and any(c.isalpha() for c in word):
                tokens = list(word)

                # For other words,
                # Apply the boundary logic
                boundaries = self._identify_morpheme_boundaries(word)

                # Add tokens and boundaries
                word_data.append((tokens, boundaries, count))

        # Free up the memory
        del word_freq
        gc.collect()

        # Apply all previous merges to catch word_data up to checkpoint state
        for i, merge in enumerate(self.merges):
            self._merge_pair(merge, word_data)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Applied {i + 1}/{len(self.merges)} merges...")

        if verbose:
            print("Word data reconstructed and caught up!")
            print(f"Resuming training from iteration {iteration + 1}")

        # BPE training loop

        while len(self.vocab) < self.vocab_size:
            # Count all pairs in single pass
            pair_counts = self._get_pairs(word_data)

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Merge the pair for all vocab
            self._merge_pair(best_pair, word_data)

            # Add to vocabulary and move to the next position in vocab
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = self.next_token_id
                self.next_token_id += 1
                self.merges.append(best_pair)

            iteration += 1

            print(f'Running iteration {iteration}, vocabulary size: {len(self.vocab)}')
            if iteration % save_interval == 0:
                self._save_checkpoint(f"training/tokenizer/checkpoints/checkpoint_{iteration}.pkl", iteration)

            if iteration % save_interval == 0:
                gc.collect()

            if verbose and iteration % save_interval == 0:
                print(f"Iteration {iteration}: Vocab size = {len(self.vocab)}, Merged pairs = {len(pair_counts)}")
                print(f"Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'")

        if verbose:
            print(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str):
        """Encode text to token IDs using greedy longest-match."""
        tokens = self._normalize_and_preprocess(text)
        encoded = []

        for token in tokens:
            # Skip if it's a protected word or already in vocab
            if token in self.vocab:
                encoded.append(self.vocab[token])
                continue

            # Greedy segmentation: find longest subwords
            word_tokens = []
            i = 0

            while i < len(token):
                # Find longest matching subword starting at position i
                longest_match = None
                longest_length = 0

                # Check all possible substrings starting at i
                for j in range(len(token), i, -1):  # Start from longest
                    substring = token[i:j]
                    if substring in self.vocab:
                        longest_match = substring
                        longest_length = j - i
                        break

                if longest_match:
                    word_tokens.append(longest_match)
                    i += longest_length
                else:
                    # Fallback to character level
                    char = token[i]
                    word_tokens.append(char if char in self.vocab else '<unk>')
                    i += 1

            # Convert to IDs
            for subword in word_tokens:
                encoded.append(self.vocab.get(subword, self.vocab['<unk>']))

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

    def _save_checkpoint(self, checkpoint_path, iteration):
        """Save training checkpoint"""
        checkpoint = {
            'vocab': self.vocab,
            'merges': self.merges,
            'next_token_id': self.next_token_id,
            'iteration': iteration,
            'vocab_size': self.vocab_size
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"  Checkpoint saved at iteration {iteration}")

    def _load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.vocab = checkpoint['vocab']
        self.merges = checkpoint['merges']
        self.next_token_id = checkpoint['next_token_id']
        self.vocab_size = checkpoint['vocab_size']

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        print(f"Current vocab size: {len(self.vocab):,}")
        print(f"Merges applied: {len(self.merges)}")
        return checkpoint['iteration']

    def save_vocab(self, path: str):
        """Save the trained tokenizer state to disk."""
        path = Path(path)

        tokenizer_state = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges,
            'next_token_id': self.next_token_id,
            'special_tokens': self.special_tokens,
            'pol_chars': list(self.pol_chars),
            'case_endings': self.case_endings,
            'protected_words': list(self.protected_words),
            'verb_endings': self.verb_endings
        }


        json_path = path.with_suffix('.json')
        pickle_path = path.with_suffix('.pkl')
        vocab_path = path.with_suffix('.vocab')

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)
        print(f"Saved tokenizer to {json_path}")

        with open(pickle_path, 'wb') as f:
            pickle.dump(tokenizer_state, f)
        print(f"Saved tokenizer to {pickle_path}")

        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, token_id in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{token_id}\n")

        print(f"Saved vocabulary to {vocab_path}")

    def load_vocab(self, path: str, format: str='json'):
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
#        self.alternations = state['alternations']

        print(f"Loaded tokenizer from {path}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    @classmethod
    def from_pretrained(cls, path: str, format:str='json'):
        """
        Class method to create a tokenizer instance from saved state.

        :param path: Path to saved tokenizer
        :param format: Load format ('json' or 'pickle')
        :return: Loaded tokenizer instance
        """
        tokenizer = cls()
        tokenizer.load_vocab(path, format)
        return tokenizer
