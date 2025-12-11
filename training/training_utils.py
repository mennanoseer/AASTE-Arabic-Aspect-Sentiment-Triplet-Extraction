import torch


def create_optimizer(model, args):
    """
    Create an AdamW optimizer with differentiated learning rates for BERT and task-specific layers.
    
    This optimizer applies different learning rates to BERT parameters and non-BERT parameters,
    and selectively applies weight decay to avoid regularizing bias and LayerNorm parameters.
    
    Args:
        model (torch.nn.Module): The model to optimize.
        args (argparse.Namespace): Arguments containing optimizer hyperparameters:
            - lr: Learning rate for non-BERT parameters
            - bert_lr: Learning rate for BERT parameters
            - l2: Weight decay coefficient
            - adam_epsilon: Epsilon for Adam optimizer
        
    Returns:
        torch.optim.AdamW: Configured optimizer with parameter groups.
    """
    # Parameters to exclude from weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    # BERT-specific parameters (embeddings and encoder layers)
    bert_params = ['bert.embeddings', 'bert.encoder']

    # Group parameters for different learning rates and weight decay settings
    optimizer_grouped_parameters = [
        {
            # BERT parameters with weight decay
            "params": [p for n, p in model.named_parameters() if
                      not any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)],
            "weight_decay": args.l2,
            "lr": args.bert_lr
        },
        {
            # BERT parameters without weight decay (bias and LayerNorm)
            "params": [p for n, p in model.named_parameters() if
                      any(nd in n for nd in no_decay) and any(nd in n for nd in bert_params)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            # Non-BERT parameters with weight decay
            "params": [p for n, p in model.named_parameters() if
                      not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_params)],
            "weight_decay": args.l2,
            "lr": args.lr
        },
        {
            # Non-BERT parameters without weight decay
            "params": [p for n, p in model.named_parameters() if
                      any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_params)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    return optimizer
