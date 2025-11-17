import argparse


def get_training_args():
    """
    Parse and return command-line arguments for training the ASTE model.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all hyperparameters and paths.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate the ASTE model')
    
    # Device and paths
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (cuda or cpu)')
    parser.add_argument('--dataset_dir', type=str, 
                       default='./datasets/ASTE-Data-V2-EMNLP2020_TRANSLATED_TO_ARABIC',
                       help='Directory containing the dataset')
    parser.add_argument('--saved_dir', type=str, default='outputs/models',
                       help='Directory to save trained models')
    parser.add_argument('--saved_file', type=str, default=None,
                       help='Path to save evaluation results')
    parser.add_argument('--pretrained_model', type=str, default='aubmindlab/bert-base-arabertv2',
                       help='Pretrained transformer model to use')
    parser.add_argument('--dataset', type=str, default='16res',
                       help='Dataset name (e.g., 14lap, 14res, 15res, 16res)')
    
    # Model configuration
    parser.add_argument('--version', type=str, default='3D', choices=['3D', '2D', '1D'],
                       help='Tagging scheme version')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--span_average', default=False, action='store_true',
                       help='Use averaging instead of summing for span representations')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=64,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_epoch', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training and evaluation')
    
    # Optimization hyperparameters
    parser.add_argument('--lr', type=float, default=0.0018470241747534772,
                       help='Learning rate for non-BERT parameters')
    parser.add_argument('--bert_lr', type=float, default=2e-5,
                       help='Learning rate for BERT parameters')
    parser.add_argument('--l2', type=float, default=0.0,
                       help='L2 regularization (weight decay) coefficient')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                       help='Epsilon for Adam optimizer')
    
    # Loss function configuration
    parser.add_argument('--with_weight', default=False, action='store_true',
                       help='Use class weights for handling class imbalance')
    
    args = parser.parse_args()
    return args


def get_prediction_args():
    """
    Parse and return command-line arguments for prediction with the ASTE model.
    
    Returns:
        argparse.Namespace: Parsed arguments containing prediction configuration.
    """
    parser = argparse.ArgumentParser(description='Load trained ASTE model and make predictions')
    
    # Model and paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file (.pkl)')
    parser.add_argument('--dataset_dir', type=str,
                       default='./datasets/ASTE-Data-V2-EMNLP2020_TRANSLATED_TO_ARABIC',
                       help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='16res',
                       help='Dataset name (e.g., 14lap, 14res, 15res, 16res)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save predictions as JSON (optional)')
    parser.add_argument('--pretrained_model', type=str, default='aubmindlab/bert-base-arabertv2',
                       help='Pretrained transformer model to use')
    
    # Model configuration
    parser.add_argument('--version', type=str, default='3D', choices=['3D', '2D', '1D'],
                       help='Tagging scheme version')
    
    # Evaluation configuration
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for prediction (cuda or cpu)')
    
    args = parser.parse_args()
    return args
