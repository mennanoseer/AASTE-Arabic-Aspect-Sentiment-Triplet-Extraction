import torch
from tagging.mwis_solver import MWISIntervalSolver

def extract_triplets_from_tags(tag_table, id_to_sentiment, version='3D', use_optimal=True, logits=None):
    """
    Decodes a tag table from the model into a list of aspect-sentiment-opinion triplets.

    This function implements an optimal inference algorithm using Maximum Weight Independent Set
    (MWIS) on interval graphs to find the highest-scoring set of non-overlapping spans.
    This replaces the previous greedy heuristic with a provably optimal solution.

    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.
        use_optimal (bool, optional): If True, uses MWIS optimal algorithm. If False, uses legacy greedy. Defaults to True.
        logits (torch.Tensor, optional): Raw logits from the model for scoring spans. Shape: (seq_len, seq_len, num_classes).

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    tag_table_tensor = torch.tensor(tag_table)

    # Step 1: Decode the bitmask tags into separate boolean masks for aspects, opinions, and sentiments.
    if version == '1D':  # {N, NEG, NEU, POS, O, A}
        is_aspect_span = (tag_table_tensor == 5) > 0
        is_opinion_span = (tag_table_tensor == 4) > 0
        sentiment_grid = tag_table_tensor * ((tag_table_tensor > 0) & (tag_table_tensor < 4))
    else:  # 3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
        is_aspect_span = (tag_table_tensor & 8) > 0
        is_opinion_span = (tag_table_tensor & 4) > 0
        sentiment_grid = (tag_table_tensor & 3)

    # Step 2: Identify all spans that have a sentiment label. These are our candidate regions for triplets.
    sentiment_span_indices = sentiment_grid.nonzero()
    sentiment_values = sentiment_grid[sentiment_span_indices[:, 0], sentiment_span_indices[:, 1]].unsqueeze(dim=-1)
    
    # Create a list of candidate regions [start, end, sentiment_id, region_size]
    candidate_regions = torch.cat([
        sentiment_span_indices,
        sentiment_values,
        sentiment_span_indices.sum(dim=-1, keepdim=True)
    ], dim=-1).tolist()
    
    # Sort regions by size, then by start index. This processes smaller spans first.
    candidate_regions.sort(key=lambda x: (x[-1], x[0]))

    # Step 3: Iterate through candidate regions to extract triplets.
    valid_triplets = []
    valid_triplets_set = set()
    
    # Initialize MWIS solver for optimal span selection
    mwis_solver = MWISIntervalSolver() if use_optimal else None
    
    # Compute span scores for aspects and opinions if logits are provided
    aspect_scores = None
    opinion_scores = None
    if use_optimal and logits is not None:
        aspect_scores, opinion_scores = compute_span_scores(logits, version)

    for start_idx, end_idx, sentiment_id, _ in candidate_regions:
        
        # CASE 1: Aspect-Opinion Order
        # Find all aspect spans within the current region's row.
        if use_optimal:
            aspect_candidates = extract_optimal_spans_from_region(
                is_aspect_span[start_idx, start_idx:end_idx + 1], 
                start_idx, 
                mwis_solver,
                aspect_scores[start_idx, start_idx:end_idx + 1] if aspect_scores is not None else None
            )
        else:
            aspect_candidates = find_sub_spans(is_aspect_span[start_idx, start_idx:end_idx + 1], start_idx)
        
        # Find all opinion spans within the current region's column.
        if use_optimal:
            opinion_candidates = extract_optimal_spans_from_region(
                is_opinion_span[start_idx:end_idx + 1, end_idx], 
                start_idx, 
                mwis_solver,
                opinion_scores[start_idx:end_idx + 1, end_idx] if opinion_scores is not None else None
            )
        else:
            opinion_candidates = find_sub_spans(is_opinion_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # Select the best aspect and opinion spans
            if use_optimal:
                # For optimal mode, we already have the best non-overlapping spans
                # Select the last aspect and first opinion (or apply other selection strategy)
                aspect_choice_idx = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != end_idx) else -2
                opinion_choice_idx = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != start_idx) else 1
            else:
                # Legacy greedy heuristic
                aspect_choice_idx = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != end_idx) else -2
                opinion_choice_idx = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != start_idx) else 1
            
            aspect_span = [start_idx, aspect_candidates[aspect_choice_idx]]
            opinion_span = [opinion_candidates[opinion_choice_idx], end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

        # CASE 2: Opinion-Aspect Order
        # Find all opinion spans within the current region's row.
        if use_optimal:
            opinion_candidates = extract_optimal_spans_from_region(
                is_opinion_span[start_idx, start_idx:end_idx + 1], 
                start_idx, 
                mwis_solver,
                opinion_scores[start_idx, start_idx:end_idx + 1] if opinion_scores is not None else None
            )
        else:
            opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
        
        # Find all aspect spans within the current region's column.
        if use_optimal:
            aspect_candidates = extract_optimal_spans_from_region(
                is_aspect_span[start_idx:end_idx + 1, end_idx], 
                start_idx, 
                mwis_solver,
                aspect_scores[start_idx:end_idx + 1, end_idx] if aspect_scores is not None else None
            )
        else:
            aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # Select the best opinion and aspect spans
            if use_optimal:
                opinion_choice_idx = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != end_idx) else -2
                aspect_choice_idx = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != start_idx) else 1
            else:
                # Legacy greedy heuristic
                opinion_choice_idx = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != end_idx) else -2
                aspect_choice_idx = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != start_idx) else 1
            
            opinion_span = [start_idx, opinion_candidates[opinion_choice_idx]]
            aspect_span = [aspect_candidates[aspect_choice_idx], end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

    # Step 4: Return all found entities and the final sorted triplets.
    return {
        'aspects': is_aspect_span.nonzero().squeeze().tolist(),
        'opinions': is_opinion_span.nonzero().squeeze().tolist(),
        'triplets': sorted(valid_triplets, key=lambda x: (x[0][0], x[0][-1], x[1][0], x[1][-1]))
    }

def extract_optimal_spans_from_region(span_mask, offset, mwis_solver, scores=None):
    """
    Extract optimal non-overlapping span indices from a boolean mask using MWIS.
    
    Args:
        span_mask: Boolean tensor indicating where spans are present
        offset: Offset to add to indices
        mwis_solver: Instance of MWISIntervalSolver
        scores: Optional tensor of scores for each position
    
    Returns:
        List of indices where optimal spans are located
    """
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    indices = ensure_list(indices)
    
    if not indices:
        return []
    
    # For a 1D sequence, each index represents a single-token span
    # Create spans (each position is a span of length 1)
    spans = [(idx, idx) for idx in indices]
    
    # Use scores if provided, otherwise uniform weights
    if scores is not None:
        # Extract scores for the positions that are True in the mask
        mask_indices = span_mask.nonzero().squeeze()
        if mask_indices.dim() == 0:
            mask_indices = mask_indices.unsqueeze(0)
        weights = [scores[idx].item() if idx < len(scores) else 1.0 for idx in mask_indices]
    else:
        weights = [1.0] * len(spans)
    
    # Solve MWIS
    selected_indices = mwis_solver.solve(spans, weights)
    result = [spans[idx][0] for idx in selected_indices]
    
    return result

def compute_span_scores(logits, version='3D'):
    """
    Compute scores for aspect and opinion spans from model logits.
    
    Args:
        logits: Model output logits of shape (seq_len, seq_len, num_classes)
        version: Encoding version ('1D' or '3D')
    
    Returns:
        Tuple of (aspect_scores, opinion_scores) tensors
    """
    if version == '1D':
        # For 1D version: {N, NEG, NEU, POS, O, A}
        # Aspect is class 5, Opinion is class 4
        aspect_scores = logits[:, :, 5]
        opinion_scores = logits[:, :, 4]
    else:  # 3D version
        # For 3D version, aspect is indicated by bit 8, opinion by bit 4
        # We need to sum probabilities of all classes where the aspect/opinion bit is set
        # After softmax, we want the probability mass on aspect/opinion classes
        probs = torch.softmax(logits, dim=-1)
        
        # For 3D: classes are encoded as A-O-Sentiment combinations
        # We need to identify which classes correspond to aspects and opinions
        num_classes = logits.shape[-1]
        
        # Create masks for aspect and opinion classes
        # In 3D encoding: 8 aspect classes (bit 8 set) and 8 opinion classes (bit 4 set)
        aspect_mask = torch.tensor([(i & 8) > 0 for i in range(num_classes)], dtype=torch.float32, device=logits.device)
        opinion_mask = torch.tensor([(i & 4) > 0 for i in range(num_classes)], dtype=torch.float32, device=logits.device)
        
        # Sum probabilities for aspect and opinion classes
        aspect_scores = (probs * aspect_mask).sum(dim=-1)
        opinion_scores = (probs * opinion_mask).sum(dim=-1)
    
    return aspect_scores, opinion_scores

def find_sub_spans(span_mask, offset):
    """Helper to find and return indices of true values in a boolean mask."""
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    return ensure_list(indices)

def ensure_list(item):
    """Ensures that the returned item is always a list."""
    if not isinstance(item, list):
        return [item]
    return item
