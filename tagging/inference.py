import torch

def extract_triplets_from_tags(tag_table, id_to_sentiment, version='3D', use_beam_search=False, beam_size=5):
    """
    Decodes a tag table from the model into a list of aspect-sentiment-opinion triplets.

    This function implements either a greedy inference algorithm or beam search to find 
    the most likely triplets based on the model's output grid.

    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.
        use_beam_search (bool, optional): Whether to use beam search instead of greedy. Defaults to False.
        beam_size (int, optional): Number of hypotheses to maintain in beam search. Defaults to 5.

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    if use_beam_search:
        return extract_triplets_beam_search(tag_table, id_to_sentiment, version, beam_size)
    else:
        return extract_triplets_greedy(tag_table, id_to_sentiment, version)

def extract_triplets_greedy(tag_table, id_to_sentiment, version='3D'):
    """
    Original greedy inference algorithm for triplet extraction.
    
    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.

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
    
    # Sort regions by size, then by start index. This greedy approach processes smaller spans first.
    candidate_regions.sort(key=lambda x: (x[-1], x[0]))

    # Step 3: Iterate through candidate regions to extract triplets.
    valid_triplets = []
    valid_triplets_set = set()

    for start_idx, end_idx, sentiment_id, _ in candidate_regions:
        
        # CASE 1: Aspect-Opinion Order
        # Find all aspect spans within the current region's row.
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx, start_idx:end_idx + 1], start_idx)
        # Find all opinion spans within the current region's column.
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # This heuristic selects the most likely aspect/opinion from the candidates.
            # It prefers spans that are not at the boundaries of the region unless it's the only option.
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
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
        # Find all aspect spans within the current region's column.
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # Apply the same selection heuristic as in Case 1.
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

def find_sub_spans(span_mask, offset):
    """Helper to find and return indices of true values in a boolean mask."""
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    return ensure_list(indices)

def ensure_list(item):
    """Ensures that the returned item is always a list."""
    if not isinstance(item, list):
        return [item]
    return item


def extract_triplets_beam_search(tag_table, id_to_sentiment, version='3D', beam_size=5):
    """
    Beam search inference algorithm for triplet extraction.
    
    Explores multiple hypotheses simultaneously to find better triplet combinations.
    
    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string.
        version (str): The encoding version of the tags ('1D' or '3D').
        beam_size (int): Number of best hypotheses to maintain.

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    tag_table_tensor = torch.tensor(tag_table)
    
    # Step 1: Decode the bitmask tags
    if version == '1D':
        is_aspect_span = (tag_table_tensor == 5) > 0
        is_opinion_span = (tag_table_tensor == 4) > 0
        sentiment_grid = tag_table_tensor * ((tag_table_tensor > 0) & (tag_table_tensor < 4))
    else:  # 3D
        is_aspect_span = (tag_table_tensor & 8) > 0
        is_opinion_span = (tag_table_tensor & 4) > 0
        sentiment_grid = (tag_table_tensor & 3)
    
    # Step 2: Get all candidate regions with sentiment
    sentiment_span_indices = sentiment_grid.nonzero()
    sentiment_values = sentiment_grid[sentiment_span_indices[:, 0], sentiment_span_indices[:, 1]].unsqueeze(dim=-1)
    
    candidate_regions = torch.cat([
        sentiment_span_indices,
        sentiment_values,
        sentiment_span_indices.sum(dim=-1, keepdim=True)
    ], dim=-1).tolist()
    
    # Sort by size, then start index
    candidate_regions.sort(key=lambda x: (x[-1], x[0]))
    
    # Step 3: Generate all possible triplets for each region
    all_candidate_triplets = []
    
    for start_idx, end_idx, sentiment_id, _ in candidate_regions:
        # Extract possible triplets for this region
        region_triplets = _extract_region_triplets(
            start_idx, end_idx, sentiment_id, 
            is_aspect_span, is_opinion_span, 
            id_to_sentiment
        )
        all_candidate_triplets.extend(region_triplets)
    
    # Step 4: Beam search to find best non-overlapping combination
    best_triplets = beam_search_triplet_combination(all_candidate_triplets, beam_size)
    
    return {
        'aspects': is_aspect_span.nonzero().squeeze().tolist(),
        'opinions': is_opinion_span.nonzero().squeeze().tolist(),
        'triplets': sorted(best_triplets, key=lambda x: (x[0][0], x[0][-1], x[1][0], x[1][-1]))
    }


def _extract_region_triplets(start_idx, end_idx, sentiment_id, 
                             is_aspect_span, is_opinion_span, id_to_sentiment):
    """
    Extract all possible triplets from a sentiment region.
    
    Args:
        start_idx (int): Start index of the sentiment region.
        end_idx (int): End index of the sentiment region.
        sentiment_id (int): Sentiment ID for this region.
        is_aspect_span (torch.Tensor): Boolean mask for aspect spans.
        is_opinion_span (torch.Tensor): Boolean mask for opinion spans.
        id_to_sentiment (dict): Mapping from sentiment ID to sentiment string.
    
    Returns:
        list: List of (triplet, score) tuples.
    """
    triplets = []
    sentiment = id_to_sentiment[sentiment_id]
    
    # CASE 1: Aspect-Opinion Order
    aspect_candidates = find_sub_spans(is_aspect_span[start_idx, start_idx:end_idx + 1], start_idx)
    opinion_candidates = find_sub_spans(is_opinion_span[start_idx:end_idx + 1, end_idx], start_idx)
    
    if aspect_candidates and opinion_candidates:
        for aspect_end in aspect_candidates:
            for opinion_start in opinion_candidates:
                aspect_span = (start_idx, aspect_end)
                opinion_span = (opinion_start, end_idx)
                score = _compute_triplet_score(aspect_span, opinion_span, start_idx, end_idx)
                triplets.append(((aspect_span, opinion_span, sentiment), score))
    
    # CASE 2: Opinion-Aspect Order
    opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
    aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)
    
    if aspect_candidates and opinion_candidates:
        for opinion_end in opinion_candidates:
            for aspect_start in aspect_candidates:
                opinion_span = (start_idx, opinion_end)
                aspect_span = (aspect_start, end_idx)
                score = _compute_triplet_score(aspect_span, opinion_span, start_idx, end_idx)
                triplets.append(((aspect_span, opinion_span, sentiment), score))
    
    return triplets


def _compute_triplet_score(aspect_span, opinion_span, region_start, region_end):
    """
    Compute a confidence score for a triplet.
    
    Higher scores for:
    - Shorter spans (more specific)
    - Spans not at region boundaries (more confident)
    - Balanced aspect/opinion lengths
    
    Args:
        aspect_span (tuple): (start, end) of aspect.
        opinion_span (tuple): (start, end) of opinion.
        region_start (int): Start of sentiment region.
        region_end (int): End of sentiment region.
    
    Returns:
        float: Score for this triplet (higher is better).
    """
    score = 1.0
    
    # Prefer shorter spans (more specific)
    aspect_length = aspect_span[1] - aspect_span[0] + 1
    opinion_length = opinion_span[1] - opinion_span[0] + 1
    score += 1.0 / (aspect_length + opinion_length)
    
    # Penalize boundary positions (less confident)
    if aspect_span[0] == region_start or aspect_span[1] == region_end:
        score -= 0.3
    if opinion_span[0] == region_start or opinion_span[1] == region_end:
        score -= 0.3
    
    # Prefer balanced aspect/opinion lengths
    length_ratio = min(aspect_length, opinion_length) / max(aspect_length, opinion_length)
    score += 0.2 * length_ratio
    
    return score


def beam_search_triplet_combination(candidate_triplets, beam_size=5):
    """
    Use beam search to find the best non-overlapping combination of triplets.
    
    Args:
        candidate_triplets (list): List of (triplet, score) tuples.
        beam_size (int): Number of hypotheses to maintain.
    
    Returns:
        list: Best combination of non-overlapping triplets.
    """
    if not candidate_triplets:
        return []
    
    # Sort candidates by score (descending)
    candidate_triplets.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize beam with empty hypothesis
    # Each hypothesis is (triplet_list, used_positions, total_score)
    beam = [([], set(), 0.0)]
    
    # Process each candidate
    for triplet, score in candidate_triplets:
        new_beam = []
        
        for current_triplets, used_positions, current_score in beam:
            # Option 1: Don't add this triplet
            new_beam.append((current_triplets, used_positions, current_score))
            
            # Option 2: Add this triplet if no overlap
            aspect_span, opinion_span, sentiment = triplet
            triplet_positions = set(range(aspect_span[0], aspect_span[1] + 1))
            triplet_positions.update(range(opinion_span[0], opinion_span[1] + 1))
            
            if not triplet_positions & used_positions:  # No overlap
                new_triplets = current_triplets + [triplet]
                new_positions = used_positions | triplet_positions
                new_score = current_score + score
                new_beam.append((new_triplets, new_positions, new_score))
        
        # Keep only top beam_size hypotheses
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_size]
    
    # Return the best hypothesis
    best_hypothesis = max(beam, key=lambda x: x[2])
    return best_hypothesis[0]

