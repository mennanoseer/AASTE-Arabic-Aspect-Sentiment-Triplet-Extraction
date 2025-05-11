import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class base_model(nn.Module):
    """
    Base model for Aspect Sentiment Triplet Extraction (ASTE) using BERT.
    
    This model encodes sentences using BERT, creates span representations for
    identifying aspect and opinion terms, and classifies their relationships.
    
    Args:
        pretrained_model_path (str): Path to pretrained BERT model
        hidden_dim (int): Dimension of hidden representations
        dropout (float): Dropout rate for regularization
        class_n (int, optional): Number of classes for classification. Defaults to 16.
        span_average (bool, optional): Whether to average span representations. Defaults to False.
    """
    def __init__(self, pretrained_model_path, hidden_dim, dropout, class_n=16, span_average=False):
        super().__init__()
        
        # Encoder
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        self.dense = nn.Linear(self.bert.pooler.dense.out_features, hidden_dim)
        self.span_average = span_average

        # Classifier
        self.classifier = nn.Linear(hidden_dim * 3, class_n)
        
        # dropout
        self.layer_drop = nn.Dropout(dropout)
        
    def forward(self, inputs, weight=None):
        """
        Forward pass of the model.
        
        Args:
            inputs (dict): Input dictionary containing:
                - bert_token: BERT token IDs
                - bert_word_mapback: Mapping from BERT tokens to original words
                - token_length: Length of original token sequence
                - bert_length: Length of BERT token sequence
                - golden_label (optional): Ground truth labels
            weight (torch.Tensor, optional): Class weights for loss calculation
            
        Returns:
            dict: Dictionary containing:
                - logits: Classification logits
                - loss: Loss value (if golden_label is provided)
        """
        
        #############################################################################################
        # Word representation
        bert_token = inputs['bert_token']
        attention_mask = (bert_token > 0).int()  # Create attention mask where tokens > 0
        bert_word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        bert_length = inputs['bert_length']
        
        # Get BERT contextual embeddings
        bert_out = self.bert(bert_token, attention_mask=attention_mask).last_hidden_state  # \hat{h}
        
        # Create sequence mask to handle variable lengths and apply to BERT output
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()  # Skip CLS token
        
        # Create one-hot mapping for BERT tokens to original words
        word_mapback_one_hot = (F.one_hot(bert_word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        
        # Map BERT token embeddings to word embeddings using the mapping
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        
        # Normalize word embeddings by the number of subwords
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)  # Avoid division by zero
        bert_out = bert_out / wnt.unsqueeze(dim=-1)  # h_i
        
        #############################################################################################
        # Span representation
        
        max_seq = bert_out.shape[1]
        
        # Create token mask and candidate span mask
        token_length_mask = sequence_mask(token_length)
        # Create upper triangular matrix to represent valid spans (i<=j)
        candidate_tag_mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.int64, device=bert_out.device), 
                                        diagonal=0).unsqueeze(dim=0) * (token_length_mask.unsqueeze(dim=1) * 
                                                                       token_length_mask.unsqueeze(dim=-1))
        
        # Create features for boundary words of spans (h_i and h_j)
        boundary_table_features = torch.cat([
            bert_out.unsqueeze(dim=2).repeat(1, 1, max_seq, 1),  # h_i (start)
            bert_out.unsqueeze(dim=1).repeat(1, max_seq, 1, 1)   # h_j (end)
        ], dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)
        
        # Create features for the entire span (sum or average of h_i to h_j)
        span_table_features = form_raw_span_features(bert_out, candidate_tag_mask, is_average=self.span_average)
        
        # Combine boundary and span features: h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features, span_table_features], dim=-1)
       
        #############################################################################################
        # Classification
        logits = self.classifier(self.layer_drop(table_features)) * candidate_tag_mask.unsqueeze(dim=-1)
        
        outputs = {
            'logits': logits
        }
        
        # Calculate loss if golden labels are provided
        if 'golden_label' in inputs and inputs['golden_label'] is not None:
            loss = calcualte_loss(logits, inputs['golden_label'], candidate_tag_mask, weight=weight)
            outputs['loss'] = loss
        
        return outputs
            
 
def sequence_mask(lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    
    Args:
        lengths (torch.Tensor): Tensor of sequence lengths
        max_len (int, optional): Maximum length. If None, uses max value in lengths.
        
    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, max_len) where True indicates valid positions
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))


def form_raw_span_features(v, candidate_tag_mask, is_average=True):
    """
    Form span features by aggregating token representations within spans.
    
    Args:
        v (torch.Tensor): Token representations of shape (batch_size, seq_len, hidden_dim)
        candidate_tag_mask (torch.Tensor): Mask indicating valid spans
        is_average (bool, optional): Whether to average span representations. Defaults to True.
        
    Returns:
        torch.Tensor: Span representations of shape (batch_size, seq_len, seq_len, hidden_dim)
    """
    # Apply mask to token representations
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    
    # Compute span representations by summing token representations within spans
    span_features = torch.matmul(new_v.transpose(1, -1).transpose(2, -1), 
                                 candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2, 1).transpose(2, -1)
    
    # Average if specified
    if is_average:
        _, max_seq, _ = v.shape
        # Calculate span lengths for averaging
        sub_v = torch.tensor(range(1, max_seq + 1), device=v.device).unsqueeze(dim=-1) - torch.tensor(
            range(max_seq), device=v.device)
        sub_v = torch.where(sub_v > 0, sub_v, 1).T
        
        # Divide by span length to get average
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)
        
    return span_features


def calcualte_loss(logits, golden_label, candidate_tag_mask, weight=None):
    """
    Calculate cross-entropy loss for span classification.
    
    Args:
        logits (torch.Tensor): Model predictions of shape (batch_size, seq_len, seq_len, class_n)
        golden_label (torch.Tensor): Ground truth labels of shape (batch_size, seq_len, seq_len)
        candidate_tag_mask (torch.Tensor): Mask for valid spans
        weight (torch.Tensor, optional): Class weights for weighted loss
        
    Returns:
        torch.Tensor: Loss value
    """
    loss_func = nn.CrossEntropyLoss(weight=weight, reduction='none')
    # Calculate loss and apply mask to only consider valid spans
    return (loss_func(logits.view(-1, logits.shape[-1]), 
                     golden_label.view(-1)
                     ).view(golden_label.size()) * candidate_tag_mask).sum()
