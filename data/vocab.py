import os
from collections import Counter
from data.data_utils import normalize_arabic_tokens

class Vocab:
    """
    A vocabulary class that maps tokens to integer indices.
    
    This class takes a token counter and builds a vocabulary, including special
    tokens for padding and unknown words. It provides mappings from index-to-string (itos)
    and string-to-index (stoi).
    
    Args:
        counter (Counter): A Counter object with token frequencies.
        specials (list, optional): A list of special tokens (e.g., <pad>, <unk>).
                                   Defaults to ['<pad>', '<unk>'].
    """
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index, self.unk_index = 0, 1
        self.itos = list(specials)
        
        # Sort words first by frequency (descending), then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda item: item[0])
        words_and_frequencies.sort(key=lambda item: item[1], reverse=True)
        
        # Add words to the index-to-string list
        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # Create the string-to-index mapping
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        
    def __len__(self):
        """Returns the total number of tokens in the vocabulary."""
        return len(self.itos)

def build_vocab_from_files(dataset_dir):
    """
    Builds a vocabulary by collecting all tokens from the dataset files.
    
    Args:
        dataset_dir (str): The path to the dataset directory.
        
    Returns:
        list: A list of all unique tokens found in the dataset.
    """
    all_tokens = []
    
    files = ['train_triplets.txt', 'dev_triplets.txt', 'test_triplets.txt']
    for file_name in files:
        file_path = os.path.join(dataset_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
        
        # Extract tokens from each line and add them to the list
        for line in lines:
            if '####' in line:
                tokens = line.strip().split('####')[0].split()
                normalized_tokens = normalize_arabic_tokens(tokens)
                all_tokens.extend(normalized_tokens)

    return all_tokens


def load_vocab(dataset_dir, lower=True):
    """
    Loads and creates a vocabulary object from the dataset.
    
    Args:
        dataset_dir (str): The path to the dataset directory.
        lower (bool): Whether to convert all tokens to lowercase.
        
    Returns:
        dict: A dictionary containing the token vocabulary object.
    """
    # Build a list of all tokens from the dataset files
    tokens = build_vocab_from_files(dataset_dir)
    
    # Convert all tokens to lowercase if specified
    if lower:
        tokens = [w.lower() for w in tokens]
    
    # Count the frequency of each token
    token_counts = Counter(tokens)
    
    # Create a Vocab object with special tokens for padding and unknown words
    token_vocab = Vocab(token_counts, specials=["<pad>", "<unk>"])
    
    # Return the vocabulary in a dictionary for consistency with other parts of the codebase
    return {'token_vocab': token_vocab}
