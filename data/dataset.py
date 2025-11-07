import torch
import numpy as np
from torch.utils.data import Dataset

from data.vocab import Vocab
from tagging.span_tagging import create_tag_table, convert_tags_to_ids
from data.data_utils import line_to_dict
from utils.helpers import get_long_tensor


class ASTE_End2End_Dataset(Dataset):
    """
    Dataset class for the Aspect Sentiment Triplet Extraction (ASTE) end-to-end model.
    
    This class processes raw text data containing aspect-sentiment-opinion triplets
    and converts them into a model-ready format.
    
    Args:
        file_path (str or list): Path to the data file or a list of preprocessed data.
        vocab (dict): Vocabulary mapping for tokens and labels.
        version (str): The version of the tagging scheme (e.g., '3D').
        tokenizer: A transformer tokenizer (e.g., from Hugging Face).
        max_len (int): Maximum sequence length for the transformer input.
        lower (bool): Whether to convert all tokens to lowercase.
        is_clean (bool): Whether to clean the data by removing duplicate triplets.
    """
    def __init__(self, file_path, vocab, version='3D', tokenizer=None, max_len=128, lower=True, is_clean=True):
        super().__init__()
        
        self.max_len = max_len
        self.lower = lower
        self.version = version
        self.tokenizer = tokenizer
        
        # Load data from a file or use the provided list directly
        if isinstance(file_path, str):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.raw_data = [line_to_dict(line, is_clean=is_clean) for line in lines]
        else:
            self.raw_data = file_path
        
        # Process the raw data into a model-ready format
        self.data = self.preprocess(self.raw_data, vocab, version)
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset at the given index."""
        return self.data[idx]
    
    def text_to_bert_ids(self, tokens):
        """
        Converts a list of word tokens to transformer-compatible input IDs and a word map.
        
        Args:
            tokens (list): A list of word tokens from a single sentence.
            
        Returns:
            tuple: A tuple containing:
                - bert_token_ids (list): The IDs for the transformer tokens.
                - token_to_word_map (list): A map from each transformer token back to its original word index.
        """
        bert_tokens = []
        token_to_word_map = []
        
        for i, word in enumerate(tokens):
            # Tokenize the word using the transformer tokenizer
            sub_tokens = self.tokenizer.tokenize(word)
            bert_tokens.extend(sub_tokens)
            # Map each sub-token back to the original word index
            token_to_word_map.extend([i] * len(sub_tokens))
            
        # Convert the token strings to their corresponding IDs
        bert_token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        return bert_token_ids, token_to_word_map
    
    def preprocess(self, data, vocab, version):
        """
        Processes raw data samples into a format suitable for model input.
        
        Args:
            data (list): A list of raw data dictionaries.
            vocab (dict): A vocabulary dictionary with token and label mappings.
            version (str): The version of the tagging scheme.
            
        Returns:
            list: A list of processed data samples.
        """
        token_vocab = vocab['token_vocab']
        label_to_id = vocab['label_vocab']['label_to_id']
        processed_samples = []
        
        # Get special token IDs from the tokenizer
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        
        for raw_sample in data:
            # Create the ground truth label table if triplets are available
            if 'triplets' in raw_sample:
                tag_table = create_tag_table(raw_sample, version=version)
                golden_label = convert_tags_to_ids(tag_table, label_to_id)
            else:
                golden_label = None
            
            # Process tokens
            tokens = raw_sample['token']
            if self.lower:
                tokens = [t.lower() for t in tokens]
            
            # Convert text to transformer token IDs and get the word map
            bert_token_ids, token_to_word_map = self.text_to_bert_ids(tokens)

            # Truncate sequences to the specified max length
            bert_token_ids = bert_token_ids[:self.max_len]
            token_to_word_map = token_to_word_map[:self.max_len]
            
            # Determine the number of original words after truncation
            word_count = token_to_word_map[-1] + 1 if token_to_word_map else 0
            assert word_count == len(tokens) or word_count < len(tokens)
            
            bert_token_count = len(token_to_word_map)
            
            # Truncate original tokens and convert to vocabulary IDs
            tokens = tokens[:word_count]
            token_ids = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tokens]
            
            # Assemble the final processed sample
            processed_sample = {
                'token_ids': token_ids,
                'word_count': word_count,
                'bert_token_ids': [cls_id] + bert_token_ids + [sep_id],
                'bert_token_count': bert_token_count,
                'token_to_word_map': token_to_word_map,
                'golden_label': golden_label
            }
            processed_samples.append(processed_sample)
            
        return processed_samples


def aste_collate_fn(batch):
    """
    Collate function to create batches from dataset samples.
    
    This function pads sequences to the same length within a batch and
    assembles the individual samples into batch tensors.
    
    Args:
        batch (list): A list of samples from the dataset.
        
    Returns:
        dict: A dictionary containing batch tensors ready for model input.
    """
    batch_size = len(batch)
    
    # Pad token sequences to the max length in the batch
    token_ids = get_long_tensor([s['token_ids'] for s in batch])
    word_counts = torch.tensor([s['word_count'] for s in batch])
    bert_token_ids = get_long_tensor([s['bert_token_ids'] for s in batch])
    bert_token_counts = torch.tensor([s['bert_token_count'] for s in batch])
    token_to_word_map = get_long_tensor([s['token_to_word_map'] for s in batch])

    # Create and pad the ground truth label tensor
    max_word_count = word_counts.max()
    golden_labels = np.zeros((batch_size, max_word_count, max_word_count), dtype=np.int64)
    
    if batch[0]['golden_label'] is not None:
        for i in range(batch_size):
            count = word_counts[i]
            golden_labels[i, :count, :count] = batch[i]['golden_label']

    golden_labels = torch.from_numpy(golden_labels)
    
    # Assemble all tensors into a final batch dictionary
    batch_dict = {
        'token': token_ids,
        'token_length': word_counts,
        'bert_token': bert_token_ids,
        'bert_length': bert_token_counts,
        'bert_word_mapback': token_to_word_map,
        'golden_label': golden_labels
    }
    
    return batch_dict
