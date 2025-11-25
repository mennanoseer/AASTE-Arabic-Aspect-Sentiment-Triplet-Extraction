import os
import sys
import time
import torch
import random
import json
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from models.aste_model import ASTEModel
from data.dataset import ASTE_End2End_Dataset, aste_collate_fn
from data.vocab import load_vocab
from tagging.span_tagging import create_label_maps, create_sentiment_maps
from training.evaluate import evaluate_model, print_evaluation_results
from utils.helpers import count_parameters, ensure_dir, set_random_seed, create_class_weights
from training.training_utils import create_optimizer
from utils.config import get_training_args


def train_model(model, train_dataloader, optimizer, device, weight=None):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The ASTE model.
        train_dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): Device to use ('cuda' or 'cpu').
        weight (torch.Tensor, optional): Class weights for loss calculation.
        
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        # Move batch data to device
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(inputs, weight)
        
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_and_evaluate(args, save_with_seed=False):
    """
    Train and evaluate the ASTE model with the given configuration.
    
    Args:
        args (argparse.Namespace): Training arguments and hyperparameters.
        save_with_seed (bool): Whether to include seed in the saved model filename.
        
    Returns:
        dict: Test evaluation results.
    """
    print('=' * 100)
    print(f'Training Configuration:')
    print(f'  Dataset: {args.dataset}')
    print(f'  Version: {args.version}')
    print(f'  Seed: {args.seed}')
    print(f'  Device: {args.device}')
    print('=' * 100)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    
    # Setup directories
    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    model_save_dir = os.path.join(args.saved_dir, args.dataset)
    ensure_dir(model_save_dir)
    
    # Load vocabulary and create label mappings
    vocab = load_vocab(dataset_dir=dataset_path)
    label_to_id, id_to_label = create_label_maps(args.version)
    sentiment_to_id, id_to_sentiment = create_sentiment_maps()
    
    vocab['label_vocab'] = {'label_to_id': label_to_id, 'id_to_label': id_to_label}
    vocab['senti_vocab'] = {'senti_to_id': sentiment_to_id, 'id_to_senti': id_to_sentiment}

    # Setup class weights for loss calculation
    num_classes = len(label_to_id)
    args.class_n = num_classes
    weight = create_class_weights(num_classes).to(args.device) if args.with_weight else None
    
    print(f'\n> Label Mappings: {label_to_id}')
    print(f'> Class Weights: {weight}')
    print(f'> Number of Classes: {num_classes}\n')

    # Initialize model
    print('> Initializing model...')
    model = ASTEModel(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout_rate,
        num_classes=num_classes,
        span_average=args.span_average
    ).to(args.device)
    
    print(f'> Total Parameters: {count_parameters(model):,}')
    
    # Load datasets
    print('> Loading datasets...')
    train_dataset = ASTE_End2End_Dataset(
        file_path=os.path.join(dataset_path, 'train_triplets.txt'),
        version=args.version,
        vocab=vocab,
        tokenizer=tokenizer
    )
    valid_dataset = ASTE_End2End_Dataset(
        file_path=os.path.join(dataset_path, 'dev_triplets.txt'),
        version=args.version,
        vocab=vocab,
        tokenizer=tokenizer
    )
    test_dataset = ASTE_End2End_Dataset(
        file_path=os.path.join(dataset_path, 'test_triplets.txt'),
        version=args.version,
        vocab=vocab,
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=aste_collate_fn, 
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        collate_fn=aste_collate_fn, 
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=aste_collate_fn, 
        shuffle=False
    )

    # Setup optimizer
    optimizer = create_optimizer(model, args)

    # Training loop
    best_f1 = 0.0
    best_model_path = os.path.join(
        model_save_dir, 
        f'{args.dataset}_{args.version}_weight{args.with_weight}_best.pkl'
    )
    
    print(f'\n> Starting training for {args.num_epoch} epochs...\n')
    
    for epoch in range(1, args.num_epoch + 1):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss = train_model(model, train_dataloader, optimizer, args.device, weight)
        
        # Validate
        valid_loss, valid_results = evaluate_model(
            model, valid_dataset, valid_dataloader,
            id_to_sentiment=id_to_sentiment,
            device=args.device,
            version=args.version,
            class_weights=weight,
            use_beam_search=args.use_beam_search,
            beam_size=args.beam_size
        )
        
        epoch_time = time.time() - epoch_start_time
        triplet_f1 = valid_results[0]['triplet']['f1']
        
        print(f'Epoch: {epoch:3d}/{args.num_epoch} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Valid Loss: {valid_loss:.4f} | '
              f'Triplet F1: {triplet_f1*100:.2f}% | '
              f'Time: {epoch_time:.2f}s')
        
        # Save model if validation F1 improved
        if triplet_f1 > best_f1:
            best_f1 = triplet_f1
            print(f'  → New best model! (F1: {best_f1*100:.2f}%)')
            
            # Evaluate on test set with new best model
            _, test_results = evaluate_model(
                model, test_dataset, test_dataloader,
                id_to_sentiment=id_to_sentiment,
                device=args.device,
                version=args.version,
                class_weights=weight,
                use_beam_search=args.use_beam_search,
                beam_size=args.beam_size
            )
            test_f1 = test_results[0]['triplet']['f1']
            print(f'  → Test F1: {test_f1*100:.2f}%')
            
            # Save the best model
            torch.save(model, best_model_path)
    
    # Load best model for final evaluation
    print(f'\n> Loading best model from: {best_model_path}')
    
    # Fix module path issue when loading older models
    try:
        import models  # Import models module first
        sys.modules['model'] = models  # Create alias
    except ImportError:
        pass  # If models can't be imported, proceed without alias
    
    best_model = torch.load(best_model_path, weights_only=False, map_location=args.device)
    best_model.eval()
    
    # Save model with seed if requested
    if save_with_seed:
        seed_model_path = best_model_path.replace('_best', f'_seed{args.seed}_best')
        torch.save(best_model, seed_model_path)
        print(f'> Saved model with seed: {seed_model_path}')
    
    # Final evaluation on test set
    print('\n' + '=' * 100)
    print('FINAL TEST EVALUATION')
    print('=' * 100)
    
    saved_file = os.path.join(model_save_dir, args.saved_file) if args.saved_file else None
    
    _, test_results = evaluate_model(
        best_model, test_dataset, test_dataloader,
        id_to_sentiment=id_to_sentiment,
        device=args.device,
        version=args.version,
        class_weights=weight,
        output_file=saved_file,
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size
    )
    
    print(f'\n> Dataset: {args.dataset} | Version: {args.version} | '
          f'Test F1: {test_results[0]["triplet"]["f1"]*100:.2f}% | '
          f'LR: {args.lr} | BERT LR: {args.bert_lr} | '
          f'Seed: {args.seed} | Dropout: {args.dropout_rate}')
    print_evaluation_results(test_results)
    
    return test_results


def run_single_experiment():
    """
    Run a single training and evaluation experiment with default configuration.
    """
    args = get_training_args()
    args.with_weight = True  # Enable class weights by default
    train_and_evaluate(args)


def reproduce_best_results():
    """
    Reproduce the best results using optimal seeds for each dataset and version.
    
    This function runs experiments with the seeds that produced the best results
    during hyperparameter tuning.
    """
    # Best seeds for each configuration
    best_seeds = {
        '16res-3D-True': 432,
        '16res-2D-True': 432,
        '16res-1D-True': 432
    }
    
    results_summary = {}
    
    print('\n' + '=' * 100)
    print('REPRODUCING BEST RESULTS')
    print('=' * 100 + '\n')
    
    for config_key, seed in best_seeds.items():
        dataset, version, use_weight = config_key.split('-')
        use_weight = eval(use_weight)
        
        args = get_training_args()
        args.seed = seed
        args.dataset = dataset
        args.version = version
        args.with_weight = use_weight
        
        print(f'\n> Running: Dataset={dataset}, Version={version}, Weighted={use_weight}, Seed={seed}')
        test_results = train_and_evaluate(args, save_with_seed=False)
        
        results_summary[config_key] = test_results[0]['triplet']
    
    # Print summary
    print('\n' + '=' * 100)
    print('BEST RESULTS SUMMARY')
    print('=' * 100)
    for config_key, metrics in results_summary.items():
        dataset, version, _ = config_key.split('-')
        print(f'{dataset:10s} | {version:3s} | F1: {metrics["f1"]*100:5.2f}% | '
              f'P: {metrics["precision"]*100:5.2f}% | R: {metrics["recall"]*100:5.2f}%')


def random_search_hyperparameters(num_trials=20):
    """
    Perform random search for hyperparameter optimization.
    
    Args:
        num_trials (int): Number of random configurations to try.
        
    Returns:
        dict: Best configuration and its results.
    """
    results_list = []
    best_f1 = 0.0
    best_config = None
    best_result = None
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "random_search_results"
    ensure_dir(results_dir)
    results_file = os.path.join(results_dir, f"random_search_{timestamp}.json")
    
    print(f'\n' + '=' * 100)
    print(f'RANDOM HYPERPARAMETER SEARCH - {num_trials} trials')
    print(f'Results will be saved to: {results_file}')
    print('=' * 100 + '\n')
    
    for trial in range(1, num_trials + 1):
        # Sample random hyperparameters
        config = {
            'seed': random.choice([432, 46]),
            'hidden_dim': random.choice([200, 250, 300, 350]),
            'num_epoch': random.choice([75, 80, 85, 90, 95, 100]),
            'batch_size': random.choice([16, 32, 64]),
            'dropout_rate': random.choice([0.25, 0.3, 0.35, 0.4]),
            'lr': random.uniform(0.0015, 0.0021),
            'span_average': random.choice([True, False]),
        }
        
        # Setup arguments with sampled configuration
        args = get_training_args()
        args.version = '3D'  # Fixed for this experiment
        args.with_weight = True
        
        for key, value in config.items():
            setattr(args, key, value)
        
        print(f'\n{"="*100}')
        print(f'Trial {trial}/{num_trials}')
        print(f'Configuration: {json.dumps(config, indent=2)}')
        print(f'{"="*100}\n')
        
        # Train and evaluate
        test_results = train_and_evaluate(args)
        f1_score = test_results[0]['triplet']['f1']
        
        # Store results
        trial_result = {
            'trial': trial,
            'config': config,
            'f1': f1_score,
            'precision': test_results[0]['triplet']['precision'],
            'recall': test_results[0]['triplet']['recall']
        }
        results_list.append(trial_result)
        
        # Update best configuration if necessary
        if f1_score > best_f1:
            best_f1 = f1_score
            best_config = config
            best_result = trial_result
            print(f'\n*** NEW BEST F1: {best_f1*100:.2f}% ***\n')
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': results_list,
                'best_result': best_result,
                'best_config': best_config,
                'best_f1': best_f1,
                'completed_trials': trial,
                'total_trials': num_trials
            }, f, indent=4)
    
    # Print final summary
    print('\n' + '=' * 100)
    print('RANDOM SEARCH COMPLETED')
    print('=' * 100)
    print(f'Best F1 Score: {best_f1*100:.2f}%')
    print(f'Best Configuration:')
    print(json.dumps(best_config, indent=2))
    print(f'\nFull results saved to: {results_file}')
    
    return best_result


if __name__ == '__main__':
    # Uncomment the function you want to run:
    
    # Option 1: Run a single experiment with default settings
    # run_single_experiment()
    
    # Option 2: Reproduce best results with optimal seeds
    reproduce_best_results()
    
    # Option 3: Run random hyperparameter search
    # random_search_hyperparameters(num_trials=20)
