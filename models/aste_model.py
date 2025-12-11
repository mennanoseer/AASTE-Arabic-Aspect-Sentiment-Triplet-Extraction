import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ASTEModel(nn.Module):
    """
    Aspect Sentiment Triplet Extraction (ASTE) model.

    This model uses a pretrained transformer encoder (like BERT) to extract
    aspect-sentiment triplets from text. It identifies spans for aspects and
    opinions and classifies the sentiment relationship between them.

    Args:
        pretrained_model_name (str): Name or path to a pretrained transformer model.
        hidden_dim (int): The dimension of the hidden representations.
        dropout (float): The dropout rate for regularization.
        num_classes (int, optional): The number of classes for span classification. Defaults to 16.
        span_average (bool, optional): If True, averages token embeddings for span representations. 
                                     If False, it sums them. Defaults to False.
    """
    def __init__(self, pretrained_model_name, hidden_dim, dropout, num_classes=16, span_average=False):
        super().__init__()
        
        # Encoder
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.span_average = span_average

        # Classifier
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, weight=None):
        """
        Forward pass of the ASTE model.

        Args:
            inputs (dict): A dictionary of input tensors containing:
                - 'bert_token': Input token IDs for the transformer model.
                - 'bert_word_mapback': A map to reconstruct word embeddings from token embeddings.
                - 'token_length': The length of the original word sequence.
                - 'bert_length': The length of the transformer token sequence.
                - 'golden_label' (optional): Ground truth labels for loss calculation.
            weight (torch.Tensor, optional): Class weights for the loss function.

        Returns:
            dict: A dictionary with 'logits' and optionally 'loss'.
        """
        
        # Unpack inputs
        input_ids = inputs['bert_token']
        word_map = inputs['bert_word_mapback']
        word_lengths = inputs['token_length']
        token_lengths = inputs['bert_length']
        
        #############################################################################################
        # 1. Word-level Representation
        #############################################################################################
        
        attention_mask = (input_ids > 0).int()
        
        # Get contextual token embeddings from the transformer
        token_embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Create a mask for valid tokens (excluding padding)
        token_mask = sequence_mask(token_lengths).unsqueeze(dim=-1)
        # Trim the [CLS] token and apply the mask
        token_embeddings = token_embeddings[:, 1:max(token_lengths) + 1, :] * token_mask.float()
        
        # Create a one-hot mapping from tokens back to words
        word_map_one_hot = (F.one_hot(word_map).float() * token_mask.float()).transpose(1, 2)
        
        # Aggregate token embeddings to get word embeddings
        word_embeddings = torch.bmm(word_map_one_hot.float(), self.dense(token_embeddings))
        
        # Normalize word embeddings by the number of subword tokens
        subword_counts = word_map_one_hot.sum(dim=-1)
        subword_counts.masked_fill_(subword_counts == 0, 1)  # Avoid division by zero
        word_embeddings = word_embeddings / subword_counts.unsqueeze(dim=-1)
        
        #############################################################################################
        # 2. Span Representation
        #############################################################################################
        
        max_word_len = word_embeddings.shape[1]
        
        # Create a mask for valid words in the sequence
        word_mask = sequence_mask(word_lengths)
        # Create a mask for all valid candidate spans (i <= j)
        span_mask = torch.triu(torch.ones(max_word_len, max_word_len, dtype=torch.int64, device=word_embeddings.device), 
                               diagonal=0).unsqueeze(dim=0) * (word_mask.unsqueeze(dim=1) * 
                                                              word_mask.unsqueeze(dim=-1))

        # Get representations for span boundaries (start and end words)
        span_start_embeddings = word_embeddings.unsqueeze(dim=2).expand(-1, -1, max_word_len, -1)
        span_end_embeddings = word_embeddings.unsqueeze(dim=1).expand(-1, max_word_len, -1, -1)
        
        # Get aggregated representation for the content within the span
        span_content_embeddings = form_raw_span_features(word_embeddings, span_mask, is_average=self.span_average)
        
        # Concatenate features to form the final span representation
        table_features = torch.cat([span_start_embeddings, span_end_embeddings, span_content_embeddings], dim=-1)
        
        #############################################################################################
        # 3. Classification
        #############################################################################################
        
        # Classify each span
        logits = self.classifier(self.dropout(table_features))
        
        # Mask out invalid spans
        logits = logits * span_mask.unsqueeze(dim=-1)
        
        outputs = {'logits': logits}
        
        # Calculate loss if ground truth labels are provided
        if 'golden_label' in inputs:
            outputs['loss'] = calculate_loss(logits, inputs['golden_label'], span_mask, weight)
            
        return outputs
 
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from a tensor of sequence lengths.
    
    Args:
        lengths (torch.Tensor): A tensor containing the length of each sequence in the batch.
        max_len (int, optional): The maximum length for the mask. If None, it's inferred 
                                 from the max value in `lengths`.
        
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, max_len) where `True` indicates
                      a valid position (i.e., part of the sequence).
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))


def form_raw_span_features(v, candidate_tag_mask, is_average=True):
    """
    Constructs span features by aggregating token representations within each span.
    
    Args:
        v (torch.Tensor): Token representations of shape (batch_size, seq_len, hidden_dim).
        candidate_tag_mask (torch.Tensor): A mask indicating valid spans.
        is_average (bool, optional): If True, averages the token representations in the span. 
                                     If False, sums them. Defaults to True.
        
    Returns:
        torch.Tensor: Span representations of shape (batch_size, seq_len, seq_len, hidden_dim).
    """
    # Apply mask to token representations
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    
    # Compute span features by summing token representations within spans
    span_features = torch.matmul(new_v.transpose(1, -1).transpose(2, -1), 
                                 candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2, 1).transpose(2, -1)
    
    # Average if specified
    if is_average:
        span_len = torch.sum(candidate_tag_mask, dim=-1)
        span_len.masked_fill_(span_len==0, 1)
        span_features = span_features / span_len.unsqueeze(dim=-1)
        
    return span_features


def calculate_loss(logits, golden_label, candidate_tag_mask, weight=None):
    """
    Calculates the cross-entropy loss for span classification.
    
    Args:
        logits (torch.Tensor): The model's prediction scores of shape 
                               (batch_size, seq_len, seq_len, num_classes).
        golden_label (torch.Tensor): The ground truth labels of shape 
                                     (batch_size, seq_len, seq_len).
        candidate_tag_mask (torch.Tensor): A mask for valid spans.
        weight (torch.Tensor, optional): Class weights for handling class imbalance.
        
    Returns:
        torch.Tensor: The calculated loss value.
    """
    loss_func = nn.CrossEntropyLoss(weight=weight, reduction='none')
    # Calculate loss and apply mask to consider only valid spans
    return (loss_func(logits.view(-1, logits.shape[-1]), 
                     golden_label.view(-1)
                     ).view(golden_label.size()) * candidate_tag_mask).sum()
