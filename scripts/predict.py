"""
Prediction module for ASTE (Aspect Sentiment Triplet Extraction) model.

This module provides functions to load a trained model and make predictions
on test data, outputting evaluation metrics and optionally saving predictions.
"""

import os
import sys
import time
import torch
import argparse
from typing import Optional, Tuple

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn

from data.dataset import ASTE_End2End_Dataset, aste_collate_fn
from data.vocab import load_vocab
from tagging.span_tagging import create_label_maps, create_sentiment_maps
from training.evaluate import evaluate_model, print_evaluation_results
from utils.config import get_prediction_args
from tagging.greedy import extract_triplets_from_tags as greedy_extract_triplets


def predict_from_args(args: argparse.Namespace) -> Tuple:
    """
    Load a trained model and evaluate it on test data using parsed arguments.
    
    Args:
        args: Parsed command-line arguments containing all configuration
        
    Returns:
        Tuple of evaluation metrics (all, single-word, multi-word, etc.)
    """
    return predict(
        model_path=args.model_path,
        version=args.version,
        dataset=args.dataset,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=args.device,
        pretrained_model=args.pretrained_model,
        dataset_dir=args.dataset_dir,
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        use_greedy_file=args.use_pure_greedy
    )


def predict(
    model_path: str,
    version: str = '3D',
    dataset: str = '16res',
    output_file: Optional[str] = None,
    batch_size: int = 16,
    device: str = 'cuda',
    pretrained_model: str = 'aubmindlab/bert-base-arabertv2',
    dataset_dir: str = './datasets/ASTE-Data-V2-EMNLP2020_TRANSLATED_TO_ARABIC',
    use_beam_search: bool = True,
    beam_size: int = 5,
    use_greedy_file: bool = False
) -> Tuple:
    """
    Load a trained model and evaluate it on test data.
    
    Args:
        model_path: Path to the saved model file (.pkl)
        version: Model version ('3D', '2D', or '1D')
        dataset: Dataset name (e.g., '16res', '14lap', '14res', '15res')
        output_file: Optional path to save predictions as JSON
        batch_size: Batch size for evaluation
        device: Device to run prediction on ('cuda' or 'cpu')
        pretrained_model: Name or path of pretrained BERT model
        dataset_dir: Base directory containing dataset folders
        use_beam_search: Whether to use beam search for inference
        beam_size: Beam size for beam search (if enabled)
        
    Returns:
        Tuple of evaluation metrics (all, single-word, multi-word, etc.)
    """
    print('=' * 100)
    print('PREDICTION SETUP')
    print('=' * 100)
    print(f'Model Path: {model_path}')
    print(f'Dataset: {dataset}')
    print(f'Version: {version}')
    print(f'Device: {device}')
    print(f'Batch Size: {batch_size}')
    print(f'Beam Search: {"Enabled" if use_beam_search else "Disabled"}')
    if use_beam_search:
        print(f'Beam Size: {beam_size}')
    print('=' * 100 + '\n')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # Build vocabulary
    dataset_path = os.path.join(dataset_dir, dataset)
    vocab = load_vocab(dataset_dir=dataset_path)

    # Create label and sentiment mappings
    label_to_id, id_to_label = create_label_maps(version)
    sentiment_to_id, id_to_sentiment = create_sentiment_maps()
    
    vocab['label_vocab'] = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label
    }
    vocab['senti_vocab'] = {
        'senti_to_id': sentiment_to_id,
        'id_to_senti': id_to_sentiment
    }
    
    # Load trained model
    print(f'Loading model from: {model_path}')
    
    # Fix module path issue when loading older models
    try:
        import models  # Import models module first
        from models.aste_model import ASTEModel
        sys.modules['model'] = models  # Create alias for 'model' module
        models.base_model = ASTEModel  # Add base_model reference for compatibility
    except ImportError:
        pass  # If models can't be imported, proceed without alias
    
    model = torch.load(model_path, map_location=device)
    
    # Add missing attributes for backward compatibility
    if not hasattr(model, 'dropout'):
        model.dropout = nn.Dropout(0.5)  # Default dropout rate
        print("Added missing dropout layer for backward compatibility")
    
    model = model.to(device)
    model.eval()
    print(f'Model loaded successfully!\n')
    
    # Load test dataset
    test_file = os.path.join(dataset_path, 'test_triplets.txt')
    test_dataset = ASTE_End2End_Dataset(
        file_path=test_file,
        vocab=vocab,
        tokenizer=tokenizer
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=aste_collate_fn,
        shuffle=False
    )

    # Run evaluation
    print('=' * 100)
    print('RUNNING PREDICTION')
    print('=' * 100)
    start_time = time.time()
    
    _, test_results = evaluate_model(
        model,
        test_dataset,
        test_dataloader,
        id_to_sentiment=id_to_sentiment,
        device=device,
        version=version,
        output_file=output_file,
        use_beam_search=use_beam_search,
        beam_size=beam_size,
        use_pure_greedy=use_greedy_file
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print('=' * 100)
    print('PREDICTION RESULTS')
    print('=' * 100)
    print(f'Time: {elapsed_time:.3f}s')
    print(f'Test F1: {test_results[0]["triplet"]["f1"]*100:.2f}%\n')
    
    print_evaluation_results(test_results)
    
    if output_file:
        print(f'\nPredictions saved to: {output_file}')
    
    return test_results


if __name__ == '__main__':
    """
    Usage examples:
    
    Basic prediction (minimal arguments):
        python predict.py --model_path best_models/16res_3D_True_best.pkl
    
    Full configuration:
        python predict.py --model_path best_models/16res_3D_True_best.pkl \
                         --dataset 16res \
                         --version 3D \
                         --batch_size 16 \
                         --device cuda \
                         --output_file predictions.json
    
    Using CPU:
        python predict.py --model_path best_models/16res_3D_True_best.pkl --device cpu
    
    Different dataset:
        python predict.py --model_path best_models/14lap_3D_True_best.pkl \
                         --dataset 14lap
    """
    # Parse command-line arguments
    args = get_prediction_args()
    
    # Run prediction with parsed arguments
    predict_from_args(args)
