import os
import time
import torch
import random
import argparse
import numpy as np

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from ASTE_dataloader import ASTE_End2End_Dataset, ASTE_collate_fn, load_vocab
from scheme.span_tagging import form_label_id_map, form_sentiment_id_map
from evaluate import evaluate_model, print_evaluate_dict


def totally_parameters(model):
    """
    Calculate the total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def ensure_dir(d, verbose=True):
    """
    Ensure a directory exists, creating it if it doesn't.
    
    Args:
        d (str): Directory path
        verbose (bool): Whether to print creation message
    """
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)


def form_weight_n(n):
    """
    Form a weight tensor for class imbalance handling in loss calculation.
    
    Args:
        n (int): Number of classes
        
    Returns:
        torch.Tensor: Weight tensor for loss function
    """
    if n > 6:
        weight = torch.ones(n)
        index_range = torch.tensor(range(n))
        # Give higher weight (2) to classes where binary AND with 3 is non-zero
        weight = weight + ((index_range & 3) > 0)
    else:
        # Use fixed weights for smaller number of classes
        weight = torch.tensor([1.0, 2.0, 2.0, 2.0, 1.0, 1.0])
    
    return weight


def train_and_evaluate(model_func, args, save_specific=False):
    """
    Train and evaluate the ASTE (Aspect Sentiment Triplet Extraction) model.
    
    Args:
        model_func: Function to create the model
        args: Arguments containing training hyperparameters
        save_specific (bool): Whether to save model with seed in filename
        
    Returns:
        dict: Test results
    """
    print('=========================================================================================================')
    set_random_seed(args.seed)
    
    # Initialize tokenizer and paths
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    dataset_dir = args.dataset_dir + '/' + args.dataset
    saved_dir = args.saved_dir + '/' + args.dataset
    ensure_dir(saved_dir)
     
    # Load vocabulary and prepare mapping dictionaries
    vocab = load_vocab(dataset_dir=dataset_dir)
    label2id, id2label = form_label_id_map(args.version)
    senti2id, id2senti = form_sentiment_id_map()
    
    vocab['label_vocab'] = dict(label2id=label2id, id2label=id2label)
    vocab['senti_vocab'] = dict(senti2id=senti2id, id2senti=id2senti)

    # Setup class count and loss weights
    class_n = len(label2id)
    args.class_n = class_n
    weight = form_weight_n(class_n).to(args.device) if args.with_weight else None
    print('> label2id:', label2id)
    print('> weight:', weight)
    print(args)

    # Initialize model
    print('> Load model...')
    base_model = model_func(pretrained_model_path=args.pretrained_model,
                            hidden_dim=args.hidden_dim,
                            dropout=args.dropout_rate,
                            class_n=class_n,
                            span_average=args.span_average).to(args.device)
    
    print('> # parameters', totally_parameters(base_model))
    
    # Load datasets
    print('> Load dataset...')
    train_dataset = ASTE_End2End_Dataset(file_name=os.path.join(dataset_dir, 'train_triplets.txt'),
                                         version=args.version,
                                         vocab=vocab,
                                         tokenizer=tokenizer)
    valid_dataset = ASTE_End2End_Dataset(file_name=os.path.join(dataset_dir, 'dev_triplets.txt'),
                                         version=args.version,
                                         vocab=vocab,
                                         tokenizer=tokenizer)
    test_dataset = ASTE_End2End_Dataset(file_name=os.path.join(dataset_dir, 'test_triplets.txt'),
                                        version=args.version,
                                        vocab=vocab,
                                        tokenizer=tokenizer)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=ASTE_collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=ASTE_collate_fn, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=ASTE_collate_fn, shuffle=False)

    # Setup optimizer
    optimizer = get_bert_optimizer(base_model, args)

    triplet_max_f1 = 0.0  # Track best F1 score for model selection
    best_model_save_path = saved_dir + '/' + args.dataset + '_' + args.version + '_' + str(args.with_weight) + '_best.pkl'
    
    # Training loop
    print('> Training...')
    for epoch in range(1, args.num_epoch+1):
        train_loss = 0.
        total_step = 0
        
        epoch_begin = time.time()
        # Training steps
        for batch in train_dataloader:
            base_model.train()
            optimizer.zero_grad()
            
            # Move batch data to device
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            outputs = base_model(inputs, weight)
            
            loss = outputs['loss']
            total_step += 1
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Validation steps
        valid_loss, valid_results = evaluate_model(base_model, valid_dataset, valid_dataloader, 
                                                   id2senti=id2senti, 
                                                   device=args.device, 
                                                   version=args.version, 
                                                   weight=weight)
        
        print('Epoch:{}/{} \ttrain_loss:{:.4f}\tvalid_loss:{:.4f}\ttriplet_f1:{:.4f}% [{:.4f}s]'.format(
            epoch, args.num_epoch, train_loss / total_step, 
            valid_loss, 100.0 * valid_results[0]['triplet']['f1'], 
            time.time()-epoch_begin))
        
        # Save model if better F1 score is achieved
        if valid_results[0]['triplet']['f1'] > triplet_max_f1:
            triplet_max_f1 = valid_results[0]['triplet']['f1']
            
            # Also evaluate on test set when new best model is found
            evaluate_model(base_model, test_dataset, test_dataloader, 
                           id2senti=id2senti, 
                           device=args.device, 
                           version=args.version, 
                           weight=weight)
            torch.save(base_model, best_model_save_path)
    
    # Load best model for final evaluation
    saved_best_model = torch.load(best_model_save_path, weights_only=False)
    if save_specific:
        # Save model with seed in filename for reproduction experiments
        torch.save(saved_best_model, best_model_save_path.replace('_best', '_' + str(args.seed) + '_best'))
    
    saved_file = (saved_dir + '/' + args.saved_file) if args.saved_file is not None else None
    
    # Final test set evaluation
    print('> Testing...')
    _, test_results = evaluate_model(saved_best_model, test_dataset, test_dataloader, 
                                     id2senti=id2senti, 
                                     device=args.device, 
                                     version=args.version, 
                                     weight=weight,
                                     saved_file=saved_file)
    
    print('------------------------------')
    print('Dataset:{}, test_f1:{:.2f}% | version:{} lr:{} bert_lr:{} seed:{} dropout:{}'.format(
        args.dataset, test_results[0]['triplet']['f1'] * 100,
        args.version, args.lr, args.bert_lr, 
        args.seed, args.dropout_rate))
    print_evaluate_dict(test_results)
    return test_results


def get_bert_optimizer(model, args):
    """
    Create an AdamW optimizer with different learning rates for BERT and non-BERT parameters.
    
    Args:
        model: PyTorch model
        args: Arguments containing optimizer hyperparameters
        
    Returns:
        torch.optim.AdamW: Configured optimizer
    """
    # Parameters to exclude from weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    # BERT-specific parameters
    diff_part = ['bert.embeddings', 'bert.encoder']

    # Group parameters for different learning rates and weight decay
    optimizer_grouped_parameters = [
        {
            # BERT parameters with weight decay
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.bert_lr
        },
        {
            # BERT parameters without weight decay
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            # Non-BERT parameters with weight decay
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.lr
        },
        {
            # Non-BERT parameters without weight decay
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer


def set_random_seed(seed):
    """
    Set random seed for reproducibility across all random modules.
    
    Args:
        seed (int): Random seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_parameters():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str, default='./data/ASTE-Data-V2-EMNLP2020')
    parser.add_argument('--saved_dir', type=str, default='saved_models')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='aubmindlab/bert-base-arabertv2')
    parser.add_argument('--dataset', type=str, default='16res')
    
    # Model version parameter
    parser.add_argument('--version', type=str, default='3D', choices=['3D', '2D', '1D'])
    
    # Random seed
    parser.add_argument('--seed', type=int, default=64)
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    # Loss function parameters
    parser.add_argument('--with_weight', default=False, action='store_true')
    parser.add_argument('--span_average', default=False, action='store_true')
    
    args = parser.parse_args()
    
    return args


def show_results(saved_results):
    """
    Format and display results for all model versions and datasets.
    
    Args:
        saved_results (dict): Dictionary of results
    """
    all_str = ''
    for version in ['1D', '2D', '3D']:
        all_str += 'STAGE' + '-' + version + '\t'
        for dataset in ['14lap', '14res', '15res', '16res']:
            k = '{}-{}-True'.format(dataset, version)
            all_str += '|{:.2f}\t{:.2f}\t{:.2f}|\t'.format(
                saved_results[k]['precision'], saved_results[k]['recall'], saved_results[k]['f1'])
        all_str += '\n'
    print(all_str)


def run():
    """
    Run a single training and evaluation experiment.
    """
    from model import base_model
    args = get_parameters()
    args.with_weight = True  # default true here
        
    train_and_evaluate(base_model, args)
    

def for_reproduce_best_results():
    """
    Reproduce the best results using specific seeds for each dataset and model version.
    """
    from model import base_model
    # Dictionary mapping dataset-version-weighted configurations to their best seed values
    seed_list_dict = {

        '16res-3D-True': 1024,

        '16res-2D-True': 63,

        '16res-1D-True': 270
    }
    
    saved_results = {}
    # Run experiments for each configuration with its best seed
    for k, seed in seed_list_dict.items():
        dataset, version, flag = k.split('-')
        flag = eval(flag)  # Convert string 'True' to boolean
        args = get_parameters()
        
        args.seed = seed
        args.dataset = dataset
        args.version = version
        args.with_weight = flag
        
        test_results = train_and_evaluate(base_model, args, save_specific=False)
        
        saved_results[k] = test_results[0]['triplet']
    
    print(saved_results)
    print('----------------------------------------------------------------')
    # Print summary of results
    for k, r in saved_results.items():
        dataset, version, flag = k.split('-')
        print('{}\t{}\t{:.2f}%'.format(dataset, version, r['f1'] * 100))


def for_reproduce_average_results():
    """
    Reproduce average results by running 5 experiments with different seeds for each configuration.
    """
    from model import base_model
    # Dictionary mapping configurations to lists of 5 seeds each
    seed_list_dict = {
        '16res-3D-True': [1024, 2038, 1002, 244, 155],
        '16res-2D-True': [63, 159, 44, 71, 23],
        '16res-1D-True': [270, 118, 216, 25, 280]
    }
    saved_results = {}
    # Run experiments for each configuration with all its seeds
    for k, seed_list in seed_list_dict.items():
        dataset, version, flag = k.split('-')
        flag = eval(flag)  # Convert string 'True' to boolean
        args = get_parameters()
        
        args.dataset = dataset
        args.version = version
        args.with_weight = flag
        
        saved_results[k] = []
        
        # Run 5 experiments with different seeds
        for seed in seed_list:
            args.seed = seed
            test_results = train_and_evaluate(base_model, args, save_specific=True)
            
            saved_results[k].append(test_results[0]['triplet'])
    
    print(saved_results)
    print('----------------------------------------------------------------')
    # Print detailed results for each run
    for k, r_list in saved_results.items():
        dataset, version, flag = k.split('-')
        for i, r in enumerate(r_list):
            print('{}\t{}\t{}\t{:.2f}%'.format(dataset, version, i, r['f1'] * 100))


if __name__ == '__main__':
    # Uncomment one of the following lines to run different experiment modes
    # run()  # Run a single experiment
    for_reproduce_best_results()  # Run experiments with best seeds
    # for_reproduce_average_results()  # Run 5 experiments for each configuration
