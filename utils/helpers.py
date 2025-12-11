import os
import torch
import random
import numpy as np

def get_long_tensor(tokens_list, max_len=None):
    """
    Converts a list of token ID lists into a padded LongTensor.
    
    Args:
        tokens_list (list): A list of token ID lists to be padded.
        max_len (int, optional): The maximum length to pad to. If None, it uses the
                                 longest sequence in the list.
    
    Returns:
        torch.LongTensor: A padded tensor of shape (batch_size, max_len).
    """
    batch_size = len(tokens_list)
    # Determine max length from the sequences if not provided
    token_len = max(len(x) for x in tokens_list) if max_len is None else max_len
    # Initialize a tensor with zeros (for padding)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    # Fill the tensor with the actual token values
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def count_parameters(model):
    """
    Calculate the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.
        
    Returns:
        int: Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def ensure_dir(directory, verbose=True):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory.
        verbose (bool): Whether to print a message when creating the directory.
    """
    if not os.path.exists(directory):
        if verbose:
            print(f"Directory '{directory}' does not exist; creating...")
        os.makedirs(directory)


def set_random_seed(seed):
    """
    Set random seed for reproducibility across all random number generators.
    
    This function ensures that experiments are reproducible by fixing the seed
    for Python's random module, NumPy, PyTorch (CPU and CUDA), and configuring
    CUDA behavior for deterministic operations.
    
    Args:
        seed (int): The random seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def create_class_weights(num_classes):
    """
    Create weight tensor for handling class imbalance in loss calculation.
    
    Args:
        num_classes (int): Number of classes in the classification task.
        
    Returns:
        torch.Tensor: Weight tensor for loss function.
    """
    if num_classes > 6:
        weight = torch.ones(num_classes)
        index_range = torch.arange(num_classes)
        # Give higher weight (2) to classes where binary AND with 3 is non-zero
        weight = weight + ((index_range & 3) > 0).float()
    else:
        # Use fixed weights for smaller number of classes
        weight = torch.tensor([1.0, 2.0, 2.0, 2.0, 1.0, 1.0])
    
    return weight
