"""
Evaluation module for ASTE (Aspect Sentiment Triplet Extraction) model.

This module provides functions to evaluate model predictions against ground truth,
computing metrics for aspects, opinions, pairs, and complete triplets.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from tagging.inference import extract_triplets_from_tags


def evaluate_model(
    model: torch.nn.Module,
    test_dataset,
    test_dataloader,
    id_to_sentiment: Dict[int, str],
    device: str = 'cuda',
    version: str = '3D',
    class_weights: Optional[torch.Tensor] = None,
    output_file: Optional[str] = None,
    use_beam_search: bool = False,
    beam_size: int = 5
) -> Tuple[float, Tuple]:
    """
    Evaluate the ASTE model on test data.
    
    Args:
        model: ASTE model to evaluate
        test_dataset: Dataset containing test data
        test_dataloader: DataLoader for batched evaluation
        id_to_sentiment: Mapping from sentiment IDs to sentiment labels
        device: Device to run evaluation on ('cuda' or 'cpu')
        version: Model version ('3D', '2D', or '1D')
        class_weights: Optional class weights for loss calculation
        output_file: Optional file path to save predictions and gold labels
        use_beam_search: Whether to use beam search for inference
        beam_size: Beam size for beam search (if enabled)
        
    Returns:
        Tuple containing:
            - Average loss over test set
            - Evaluation metrics tuple (all, one-word, multi-word, etc.)
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    # Extract tokens and gold triplets from dataset
    tokens = [test_dataset.raw_data[idx]['token'] for idx in range(len(test_dataset.raw_data))]
    gold_triplets = [test_dataset.raw_data[idx]['triplets'] for idx in range(len(test_dataset.raw_data))]
    
    # Storage for predictions
    predicted_triplets = []
    predicted_aspects = []
    predicted_opinions = []

    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            inputs = {key: value.to(device) for key, value in batch.items()}
        
            # Forward pass
            outputs = model(inputs, class_weights)

            # Accumulate loss
            loss = outputs['loss']
            total_steps += 1
            total_loss += loss.item()

            # Get predictions from logits
            predicted_tag_ids = torch.argmax(outputs['logits'], dim=-1)
            
            # Extract triplets from predicted tags
            for idx in range(len(predicted_tag_ids)):
                predictions = extract_triplets_from_tags(
                    tag_table=predicted_tag_ids[idx].tolist(), 
                    id_to_sentiment=id_to_sentiment, 
                    version=version,
                    use_beam_search=use_beam_search,
                    beam_size=beam_size
                )
                
                predicted_triplets.append(predictions['triplets'])
                predicted_aspects.append(predictions['aspects'])
                predicted_opinions.append(predictions['opinions'])

    # Save predictions to file if requested
    if output_file is not None:
        _save_predictions_to_file(
            output_file, tokens, predicted_triplets, gold_triplets, 
            predicted_aspects, predicted_opinions
        )

    # Calculate metrics
    average_loss = total_loss / total_steps
    metrics = evaluate_predictions(
        predictions=predicted_triplets,
        gold_labels=gold_triplets,
        predicted_aspects=predicted_aspects,
        predicted_opinions=predicted_opinions
    )
    
    model.train()
    return average_loss, metrics

def evaluate_predictions(
    predictions: List[List],
    gold_labels: List[List],
    predicted_aspects: List[List],
    predicted_opinions: List[List]
) -> Tuple:
    """
    Evaluate predictions against gold labels with detailed metrics.
    
    Computes metrics for:
    - All triplets
    - Single-word spans only
    - Multi-word spans only
    - Multi-word aspects
    - Multi-word opinions
    - Aspect Term Extraction (ATE) and Opinion Term Extraction (OTE)
    
    Args:
        predictions: List of predicted triplets for each sample
        gold_labels: List of gold triplets for each sample
        predicted_aspects: List of predicted aspect spans
        predicted_opinions: List of predicted opinion spans
        
    Returns:
        Tuple of dictionaries containing scores for different evaluation types
    """
    # Initialize counters for different evaluation types
    all_counts = Counter()
    single_word_counts = Counter()
    multi_word_counts = Counter()
    multi_aspect_counts = Counter()
    multi_opinion_counts = Counter()
    ate_counts = Counter()
    ote_counts = Counter()
    
    # Evaluate each sample
    for pred, gold, pred_aspect, pred_opinion in zip(
        predictions, gold_labels, predicted_aspects, predicted_opinions
    ):
        # Overall evaluation
        all_counts = _evaluate_triplet_sample(pred, gold, all_counts)
    
        # Separate triplets by span type
        pred_single, pred_multi, pred_multi_aspect, pred_multi_opinion = (
            _separate_triplets_by_span_type(pred)
        )
        gold_single, gold_multi, gold_multi_aspect, gold_multi_opinion = (
            _separate_triplets_by_span_type(gold)
        )
        
        # Evaluate by span type
        single_word_counts = _evaluate_triplet_sample(pred_single, gold_single, single_word_counts)
        multi_word_counts = _evaluate_triplet_sample(pred_multi, gold_multi, multi_word_counts)
        multi_aspect_counts = _evaluate_triplet_sample(pred_multi_aspect, gold_multi_aspect, multi_aspect_counts)
        multi_opinion_counts = _evaluate_triplet_sample(pred_multi_opinion, gold_multi_opinion, multi_opinion_counts)
        
        # Extract unique aspects and opinions from gold triplets
        gold_aspects = [[span[0], span[1]] for span in list(set([tuple(x[0]) for x in gold]))]
        gold_opinions = [[span[0], span[1]] for span in list(set([tuple(x[1]) for x in gold]))]
        
        # Ensure predictions are in list format
        pred_aspect = _ensure_nested_list(pred_aspect)
        pred_opinion = _ensure_nested_list(pred_opinion)
        
        # Evaluate term extraction
        ate_counts = _evaluate_term_extraction(pred_aspect, gold_aspects, ate_counts)
        ote_counts = _evaluate_term_extraction(pred_opinion, gold_opinions, ote_counts)
    
    # Compute scores for all evaluation types
    all_scores = _compute_score_dict(all_counts)
    single_scores = _compute_score_dict(single_word_counts)
    multi_scores = _compute_score_dict(multi_word_counts)
    multi_aspect_scores = _compute_score_dict(multi_aspect_counts)
    multi_opinion_scores = _compute_score_dict(multi_opinion_counts)
    term_scores = _compute_term_score_dict(ate_counts, ote_counts)
    
    return (all_scores, single_scores, multi_scores, 
            multi_aspect_scores, multi_opinion_scores, term_scores)


# ==================== Core Evaluation Functions ====================

def _evaluate_triplet_sample(
    predictions: List,
    gold_labels: List,
    counts: Optional[Counter] = None
) -> Counter:
    """
    Evaluate a single sample's triplet predictions.
    
    Counts matches for:
    - Aspects (first element of triplet)
    - Opinions (second element of triplet)
    - Pairs (aspect-opinion pairs, ignoring sentiment)
    - Complete triplets (aspect-opinion-sentiment)
    
    Args:
        predictions: Predicted triplets
        gold_labels: Gold standard triplets
        counts: Existing counter to update (or create new one)
        
    Returns:
        Updated counter with evaluation counts
    """
    if counts is None:
        counts = Counter()
    
    # Extract and deduplicate aspects
    gold_aspects = list(set([tuple(x[0]) for x in gold_labels]))
    pred_aspects = list(set([tuple(x[0]) for x in predictions]))

    counts['aspect_golden'] += len(gold_aspects)
    counts['aspect_predict'] += len(pred_aspects)
    
    for prediction in pred_aspects:
        if any([prediction == actual for actual in gold_aspects]):
            counts['aspect_matched'] += 1

    # Extract and deduplicate opinions
    gold_opinions = list(set([tuple(x[1]) for x in gold_labels]))
    pred_opinions = list(set([tuple(x[1]) for x in predictions]))
    
    counts['opinion_golden'] += len(gold_opinions)
    counts['opinion_predict'] += len(pred_opinions)
    
    for prediction in pred_opinions:
        if any([prediction == actual for actual in gold_opinions]):
            counts['opinion_matched'] += 1

    # Evaluate pairs and complete triplets
    gold_triplets = [(tuple(x[0]), tuple(x[1]), x[2]) for x in gold_labels]
    pred_triplets = [(tuple(x[0]), tuple(x[1]), x[2]) for x in predictions]
    
    counts['triplet_golden'] += len(gold_triplets)
    counts['triplet_predict'] += len(pred_triplets)
    
    for prediction in pred_triplets:
        # Check pair match (ignore sentiment)
        if any([prediction[:2] == actual[:2] for actual in gold_triplets]):
            counts['pair_matched'] += 1

        # Check complete triplet match
        if any([prediction == actual for actual in gold_triplets]):
            counts['triplet_matched'] += 1

    return counts


def _evaluate_term_extraction(
    predictions: List,
    gold_labels: List,
    counts: Optional[Counter] = None
) -> Counter:
    """
    Evaluate term extraction (aspects or opinions only, without sentiment).
    
    Args:
        predictions: Predicted terms
        gold_labels: Gold standard terms
        counts: Existing counter to update
        
    Returns:
        Updated counter with term extraction counts
    """
    if counts is None:
        counts = Counter()

    counts['golden'] += len(gold_labels)
    counts['predict'] += len(predictions)
    
    for prediction in predictions:
        if any([prediction == actual for actual in gold_labels]):
            counts['matched'] += 1
    
    return counts


# ==================== Helper Functions ====================

def _separate_triplets_by_span_type(triplets: List) -> Tuple[List, List, List, List]:
    """
    Separate triplets based on span length (single-word vs multi-word).
    
    Args:
        triplets: List of triplets in format [(aspect_span, opinion_span, sentiment), ...]
                  where spans are [start_idx, end_idx]
    
    Returns:
        Tuple containing:
            - single_word_triplets: Both aspect and opinion are single words
            - multi_word_triplets: At least one span is multi-word
            - multi_aspect_triplets: Aspect is multi-word
            - multi_opinion_triplets: Opinion is multi-word
    """
    single_word_triplets = []
    multi_word_triplets = []
    multi_aspect_triplets = []
    multi_opinion_triplets = []
    
    for triplet in triplets:
        aspect_span = triplet[0]
        opinion_span = triplet[1]
        
        # Check if spans are multi-word (end_idx != start_idx)
        is_multi_aspect = aspect_span[-1] != aspect_span[0]
        is_multi_opinion = opinion_span[-1] != opinion_span[0]
        
        # Categorize triplet
        if is_multi_aspect or is_multi_opinion:
            multi_word_triplets.append(triplet)
        else:
            single_word_triplets.append(triplet)
            
        if is_multi_aspect:
            multi_aspect_triplets.append(triplet)
            
        if is_multi_opinion:
            multi_opinion_triplets.append(triplet)
    
    return (single_word_triplets, multi_word_triplets, 
            multi_aspect_triplets, multi_opinion_triplets)


def _ensure_nested_list(data: List) -> List:
    """
    Ensure data is a nested list. If it's a flat list of integers, wrap it.
    
    Args:
        data: List that may be flat or nested
        
    Returns:
        Nested list
    """
    if len(data) > 0 and isinstance(data[0], int):
        return [data]
    return data


def _save_predictions_to_file(
    filepath: str,
    tokens: List[List[str]],
    predictions: List[List],
    gold_labels: List[List],
    predicted_aspects: List[List],
    predicted_opinions: List[List]
) -> None:
    """
    Save predictions and gold labels to a JSON file for analysis.
    
    Args:
        filepath: Path to output file
        tokens: List of token sequences
        predictions: Predicted triplets
        gold_labels: Gold standard triplets
        predicted_aspects: Predicted aspect spans
        predicted_opinions: Predicted opinion spans
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        combined = [
            {
                'token': token,
                'pred': pred,
                'gold': gold,
                'pred_aspect': pred_aspect,
                'pred_opinion': pred_opinion
            }
            for token, pred, gold, pred_aspect, pred_opinion in zip(
                tokens, predictions, gold_labels, predicted_aspects, predicted_opinions
            )
        ]
        json.dump(combined, f, ensure_ascii=False, indent=2)


# ==================== Scoring Functions ====================

def _compute_f1_score(
    num_predicted: int,
    num_gold: int,
    num_matched: int
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        num_predicted: Number of predicted items
        num_gold: Number of gold items
        num_matched: Number of correctly matched items
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    precision = num_matched / num_predicted if num_predicted > 0 else 0.0
    recall = num_matched / num_gold if num_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _compute_score_dict(counts: Counter) -> Dict[str, Dict[str, float]]:
    """
    Compute scores for aspects, opinions, pairs, and triplets from counts.
    
    Args:
        counts: Counter with prediction, gold, and matched counts
        
    Returns:
        Dictionary with scores for each evaluation type
    """
    scores_aspect = _compute_f1_score(
        counts['aspect_predict'], counts['aspect_golden'], counts['aspect_matched']
    )
    scores_opinion = _compute_f1_score(
        counts['opinion_predict'], counts['opinion_golden'], counts['opinion_matched']
    )
    scores_pair = _compute_f1_score(
        counts['triplet_predict'], counts['triplet_golden'], counts['pair_matched']
    )
    scores_triplet = _compute_f1_score(
        counts['triplet_predict'], counts['triplet_golden'], counts['triplet_matched']
    )
    
    return {
        'aspect': scores_aspect,
        'opinion': scores_opinion,
        'pair': scores_pair,
        'triplet': scores_triplet
    }


def _compute_term_score_dict(
    aspect_counts: Counter,
    opinion_counts: Counter
) -> Dict[str, Dict[str, float]]:
    """
    Compute scores for term extraction (ATE and OTE).
    
    Args:
        aspect_counts: Counter for aspect term extraction
        opinion_counts: Counter for opinion term extraction
        
    Returns:
        Dictionary with ATE and OTE scores
    """
    score_ate = _compute_f1_score(
        aspect_counts['predict'], aspect_counts['golden'], aspect_counts['matched']
    )
    score_ote = _compute_f1_score(
        opinion_counts['predict'], opinion_counts['golden'], opinion_counts['matched']
    )
    
    return {'ate': score_ate, 'ote': score_ote}


# ==================== Display Functions ====================

def print_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    selected_keys: Optional[List[str]] = None
) -> None:
    """
    Print evaluation metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics (aspect, opinion, pair, triplet, etc.)
        selected_keys: Optional list of keys to display (None = all keys)
    """
    if selected_keys is None:
        selected_keys = list(metrics.keys())
    
    # Print header
    print('\t  \tP\t\tR\t\tF')
    
    # Print each metric
    for key in selected_keys:
        # Add asterisk for important metrics
        marker = '*' if key in ['aspect', 'opinion', 'triplet'] else ''
        print('{:^8}\t{:.2f}%\t{:.2f}%\t{:.2f}%'.format(
            marker + key.upper(),
            100.0 * metrics[key]['precision'],
            100.0 * metrics[key]['recall'],
            100.0 * metrics[key]['f1']
        ))


def print_evaluation_results(evaluation_metrics: Tuple) -> None:
    """
    Print comprehensive evaluation results for all metric types.
    
    Args:
        evaluation_metrics: Tuple containing (all, single-word, multi-word,
                           multi-aspect, multi-opinion, term) metrics
    """
    metric_types = ['all', 'one', 'multi', 'multi_aspect', 'multi_opinion', 'term']
    
    for idx, metrics in enumerate(evaluation_metrics):
        metric_type = metric_types[idx]
        print(f'\n[ {metric_type.upper()} ]')
        
        # Determine which metrics to display
        if metric_type in ['one', 'multi', 'multi_aspect', 'multi_opinion']:
            selected_keys = ['triplet']
        elif metric_type == 'all':
            selected_keys = ['pair', 'triplet']
        else:
            selected_keys = None
        
        print_metrics_table(metrics, selected_keys=selected_keys)


