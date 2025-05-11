import torch
import re
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

from vocab import *
from scheme.span_tagging import form_raw_table, map_raw_table_to_id


class ASTE_End2End_Dataset(Dataset):
    """
    Dataset class for Aspect Sentiment Triplet Extraction (ASTE) end-to-end model.
    
    This class prepares data for ASTE task by processing raw text data containing
    aspect-sentiment-opinion triplets and converting them to model-ready format.
    
    Args:
        file_name (str or list): Path to the data file or preprocessed data list
        vocab (dict, optional): Vocabulary mapping for tokens and labels
        version (str, optional): Version of the tagging scheme, default is '3D'
        tokenizer: BERT tokenizer for tokenization
        max_len (int, optional): Maximum sequence length for BERT input, default is 128
        lower (bool, optional): Whether to lowercase all tokens, default is True
        is_clean (bool, optional): Whether to clean the data, default is True
    """
    def __init__(self, file_name, vocab=None, version='3D', tokenizer=None, max_len=128, lower=True, is_clean=True):
        super().__init__()
        
        self.max_len = max_len  # max sequence length for BERT input
        self.lower = lower
        self.version = version
        
        # Load data from file or use provided data directly
        if type(file_name) is str:
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.raw_data = [line2dict(l, is_clean=is_clean) for l in lines]
        else:
            self.raw_data = file_name
        
        self.tokenizer = tokenizer
        # Process raw data into model-ready format
        self.data = self.preprocess(self.raw_data, vocab=vocab, version=version)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a sample from the dataset at the given index."""
        return self.data[idx]
    
    def text2bert_id(self, token):
        """
        Convert tokens to BERT input IDs with mapping information.
        
        Args:
            token (list): List of words/tokens
            
        Returns:
            tuple: (bert_token_ids, word_mappings, word_split_lengths)
                - bert_token_ids: IDs for BERT tokens
                - word_mappings: Maps each BERT token back to its original word index
                - word_split_lengths: Number of BERT tokens for each original word
        """
        re_token = []  # BERT tokens after tokenization
        word_mapback = []  # Maps each BERT token back to original word
        word_split_len = []  # Records how many BERT tokens each word is split into
        
        for idx, word in enumerate(token):
            # Tokenize each word with BERT tokenizer (may split into subwords)
            temp = self.tokenizer.tokenize(word)
            # Add all subword tokens to the list
            re_token.extend(temp)
            # Track which original word each subword belongs to
            word_mapback.extend([idx] * len(temp))
            # Record how many subwords this word was split into
            word_split_len.append(len(temp))
            
        # Convert tokens to IDs
        re_id = self.tokenizer.convert_tokens_to_ids(re_token)
        return re_id, word_mapback, word_split_len
    
    def preprocess(self, data, vocab, version):
        """
        Process raw data into model-ready format.
        
        Args:
            data (list): List of raw data dictionaries
            vocab (dict): Vocabulary dictionary containing token and label mappings
            version (str): Version of the tagging scheme
            
        Returns:
            list: Processed data samples ready for model input
        """
        token_vocab = vocab['token_vocab']
        label2id = vocab['label_vocab']['label2id']
        processed = []
        max_len = self.max_len
        
        # Get special token IDs for BERT
        CLS_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])
        SEP_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        
        for d in data:
            # Convert triplets to label IDs if they exist
            golden_label = map_raw_table_to_id(form_raw_table(d, version=version), label2id) if 'triplets' in d else None
            
            # Process tokens
            tok = d['token']
            if self.lower:
                tok = [t.lower() for t in tok]
            
            # Convert text to BERT tokens and get mapping info
            text_raw_bert_indices, word_mapback, _ = self.text2bert_id(tok)
            # Truncate to max length
            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]
            
            # Get the length of original token sequence (before BERT tokenization)
            length = word_mapback[-1] + 1
            if length == len(tok):
                pass
            else: 
                print(tok)
                assert(length == len(tok))  
            bert_length = len(word_mapback)
            
            # Truncate tokens and convert to vocab IDs
            tok = tok[:length]
            tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # Create the sample with all necessary information
            temp = {
                'token': tok,
                'token_length': length,
                'bert_token': CLS_id + text_raw_bert_indices + SEP_id,
                'bert_length': bert_length,
                'bert_word_mapback': word_mapback,
                'golden_label': golden_label
            }
            processed.append(temp)
        return processed


def ASTE_collate_fn(batch):
    """
    Collate function for creating batches from ASTE dataset samples.
    
    This function pads sequences to the same length within a batch and
    combines individual samples into batch tensors.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        dict: Dictionary containing batch tensors for model input
    """
    batch_size = len(batch)
    
    re_batch = {}
    
    # Convert token lists to padded tensors
    token = get_long_tensor([batch[i]['token'] for i in range(batch_size)])
    token_length = torch.tensor([batch[i]['token_length'] for i in range(batch_size)])
    bert_token = get_long_tensor([batch[i]['bert_token'] for i in range(batch_size)])
    bert_length = torch.tensor([batch[i]['bert_length'] for i in range(batch_size)])
    bert_word_mapback = get_long_tensor([batch[i]['bert_word_mapback'] for i in range(batch_size)])

    # Create tensor for golden labels with appropriate padding
    golden_label = np.zeros((batch_size, token_length.max(), token_length.max()), dtype=np.int64)
    
    if batch[0]['golden_label'] is not None:
        for i in range(batch_size):
            # Copy golden labels into the tensor up to the actual token length
            golden_label[i, :token_length[i], :token_length[i]] = batch[i]['golden_label']

    golden_label = torch.from_numpy(golden_label)
    
    # Combine all tensors into a batch dictionary
    re_batch = {
        'token': token,
        'token_length': token_length,
        'bert_token': bert_token,
        'bert_length': bert_length,
        'bert_word_mapback': bert_word_mapback,
        'golden_label': golden_label
    }
    
    return re_batch


def get_long_tensor(tokens_list, max_len=None):
    """
    Convert list of token lists to a padded LongTensor.
    
    Args:
        tokens_list (list): List of token ID lists to be padded
        max_len (int, optional): Maximum length to pad to. If None, uses the longest sequence.
    
    Returns:
        torch.LongTensor: Padded tensor of shape (batch_size, max_len)
    """
    batch_size = len(tokens_list)
    # Determine max length from sequences if not provided
    token_len = max(len(x) for x in tokens_list) if max_len is None else max_len
    # Initialize tensor with zeros (padding)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    # Fill in the actual token values
    for i, s in enumerate(tokens_list):
        tokens[i, :min(token_len, len(s))] = torch.LongTensor(s)[:token_len]
    return tokens


############################################################################
# data preprocess functions

def clean_data(l):
    """
    Clean a data line by removing duplicate triplets.
    
    Args:
        l (str): Input line containing tokens and triplets
        
    Returns:
        str: Cleaned line with duplicate triplets removed
    """
    # Split the line into tokens and triplets
    token, triplets = l.strip().split('####')
    # Remove duplicate triplets by converting to set and back
    temp_t = list(set([str(t) for t in eval(triplets)]))
    return token + '####' + str([eval(t) for t in temp_t]) + '\n'


def line2dict(l, is_clean=False):
    """
    Convert a data line to a dictionary format.
    
    Args:
        l (str): Input line with tokens and triplets
        is_clean (bool, optional): Whether to clean the data first
        
    Returns:
        dict: Dictionary with tokens and formatted triplets
    """
    if is_clean:
        l = clean_data(l)
    
    # Split the line into sentence and triplets
    sentence, triplets = l.strip().split('####')
    
    start_end_triplets = []
    for t in eval(triplets):
        # Convert triplet format to start-end indices for aspect and opinion terms
        start_end_triplets.append(tuple([[t[0][0], t[0][-1]], [t[1][0], t[1][-1]], t[2]]))
    
    # Sort triplets by aspect start index and opinion end index
    start_end_triplets.sort(key=lambda x: (x[0][0], x[1][-1]))
    
    return dict(token=sentence.split(' '), triplets=start_end_triplets)


#############################################################################
# vocabulary functions
def normalize_arabic(text):
    """
    Normalize Arabic text by standardizing character variations using regular expressions.
    
    This function performs common Arabic text normalization:
    - Normalizes different forms of alef (أ, إ, آ) to a standard form (ا)
    - Converts taa marbuta (ة) to haa (ه)
    - Normalizes ya (ى) to regular ya (ي)
    - Removes diacritics/harakat
    - Normalizes spacing and special characters
    
    Args:
        text (str): Arabic text to normalize
        
    Returns:
        str: Normalized Arabic text
    """

    if not text:
        return text
    
    # Normalize alef variations
    text = re.sub(r'[أإآٱ]', 'ا', text)
    
    # Normalize other letters
    text = re.sub(r'ة', 'ه', text)  # Taa marbuta to haa
    text = re.sub(r'ى', 'ي', text)  # Alef maksura to yaa
    text = re.sub(r'ؤ', 'و', text)  # Hamza on waw
    text = re.sub(r'ئ', 'ي', text)  # Hamza on yaa
    
    # Remove diacritics/harakat (fatha, damma, kasra, shadda, sukun, tanwin)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Convert Arabic numerals to English
    arabic_numbers = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
                     '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
    for ar, en in arabic_numbers.items():
        text = text.replace(ar, en)
    
    # Remove tatweel (kashida) character
    text = re.sub(r'ـ', '', text)
    
    # Normalize spaces and remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_arabic_tokens(tokens):
    """
    Apply Arabic normalization to a list of tokens.
    
    Args:
        tokens (list): List of tokens to normalize
        
    Returns:
        list: List of normalized tokens
    """
    return [normalize_arabic(token) for token in tokens]

def build_vocab(dataset):
    """
    Build vocabulary from dataset files.
    
    Args:
        dataset (str): Path to dataset directory
        
    Returns:
        list: List of all tokens found in the dataset
    """
    tokens = []
    
    files = ['train_triplets.txt', 'dev_triplets.txt', 'test_triplets.txt']
    for file_name in files:
        file_path = dataset + '/' + file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract tokens from each line and add to token list
        for l in lines:
            cur_tokens = l.strip().split('####')[0].split()
            cur_tokens = normalize_arabic_tokens(cur_tokens)
            tokens.extend(cur_tokens)

    return tokens


def load_vocab(dataset_dir, lower=True):
    """
    Load and create vocabulary from dataset.
    
    Args:
        dataset_dir (str): Path to dataset directory
        lower (bool, optional): Whether to lowercase all tokens
        
    Returns:
        dict: Dictionary containing token vocabulary
    """
    # Build vocabulary from dataset files
    tokens = build_vocab(dataset_dir)
    
    # Lowercase all tokens if specified
    if lower:
        tokens = [w.lower() for w in tokens]
    
    # Count token frequencies
    token_counter = Counter(tokens)
    
    # Create Vocab object with special tokens
    token_vocab = Vocab(token_counter, specials=["<pad>", "<unk>"])
    
    # Return vocabulary in a dictionary
    vocab = {'token_vocab': token_vocab}
    return vocab
