import torch

"""
Triplet Extraction Inference Module

This module uses beam search to extract aspect-sentiment-opinion triplets from model predictions.

Main workflow:
1. extract_triplets_from_tags() - Entry point that decodes tag table into triplets
2. extract_triplets_beam_search() - Main beam search algorithm
3. extract_region_triplets() - Generates candidate triplets for each sentiment region
4. compute_triplet_score() - Scores triplets based on span properties
5. beam_search_triplet_combination() - Finds best non-overlapping triplet combination

The beam search explores multiple hypotheses to find the best combination of triplets,
prioritizing configurations with more triplets and higher confidence scores.
"""

def extract_triplets_from_tags(tag_table, id_to_sentiment, version='3D', use_beam_search=True, beam_size=5):
    """
    Decodes a tag table from the model into a list of aspect-sentiment-opinion triplets.

    This function uses beam search to find the most likely triplets based on the model's output grid.

    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.
        use_beam_search (bool, optional): DEPRECATED - always uses beam search now. Defaults to True.
        beam_size (int, optional): Number of hypotheses to maintain in beam search. Defaults to 5.

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    return extract_triplets_beam_search(tag_table, id_to_sentiment, version, beam_size)


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
        region_triplets = extract_region_triplets(
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


def find_sub_spans(span_mask, offset):
    """Helper to find and return indices of true values in a boolean mask."""
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    return ensure_list(indices)


def ensure_list(item):
    """Ensures that the returned item is always a list."""
    if not isinstance(item, list):
        return [item]
    return item


def extract_region_triplets(start_idx, end_idx, sentiment_id, 
                             is_aspect_span, is_opinion_span, id_to_sentiment):
    """
    Extract all possible triplets from a sentiment region.
    
    This function generates candidate triplets from a sentiment-tagged region.
    It considers both aspect-opinion and opinion-aspect orderings, and gives
    higher scores to triplets that match smart heuristics (e.g., avoiding boundary positions).
    
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
        # Generate all combinations but prioritize smart choices
        for aspect_end in aspect_candidates:
            for opinion_start in opinion_candidates:
                aspect_span = (start_idx, aspect_end)
                opinion_span = (opinion_start, end_idx)
                score = compute_triplet_score(aspect_span, opinion_span, start_idx, end_idx)
                
                # Boost score if this matches the smart heuristic choice
                # (prefer non-boundary positions when multiple candidates exist)
                aspect_choice_idx = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != end_idx) else -2
                opinion_choice_idx = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != start_idx) else 1
                
                if (aspect_end == aspect_candidates[aspect_choice_idx] and 
                    opinion_start == opinion_candidates[opinion_choice_idx]):
                    score += 5.0  # Strong boost for smart choice
                
                triplets.append(((aspect_span, opinion_span, sentiment), score))
    
    # CASE 2: Opinion-Aspect Order
    opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
    aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)
    
    if aspect_candidates and opinion_candidates:
        # Generate all combinations but prioritize smart choices
        for opinion_end in opinion_candidates:
            for aspect_start in aspect_candidates:
                opinion_span = (start_idx, opinion_end)
                aspect_span = (aspect_start, end_idx)
                score = compute_triplet_score(aspect_span, opinion_span, start_idx, end_idx)
                
                # Boost score if this matches the smart heuristic choice
                opinion_choice_idx = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != end_idx) else -2
                aspect_choice_idx = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != start_idx) else 1
                
                if (opinion_end == opinion_candidates[opinion_choice_idx] and 
                    aspect_start == aspect_candidates[aspect_choice_idx]):
                    score += 5.0  # Strong boost for smart choice
                
                triplets.append(((aspect_span, opinion_span, sentiment), score))
    
    return triplets


def compute_triplet_score(aspect_span, opinion_span, region_start, region_end):
    """
    Compute a confidence score for a triplet.
    
    Higher scores are given to triplets that:
    - Have shorter spans (more specific)
    - Don't touch region boundaries (more confident)
    - Are well-contained within the sentiment region
    
    Args:
        aspect_span (tuple): (start, end) of aspect.
        opinion_span (tuple): (start, end) of opinion.
        region_start (int): Start of sentiment region.
        region_end (int): End of sentiment region.
    
    Returns:
        float: Score for this triplet (higher is better).
    """
    score = 10.0  # Base score
    
    # Compute span lengths
    aspect_length = aspect_span[1] - aspect_span[0] + 1
    opinion_length = opinion_span[1] - opinion_span[0] + 1
    region_size = region_end - region_start + 1
    
    # Strong preference for shorter spans (more specific)
    # This is the primary signal
    span_score = 5.0 / (aspect_length + opinion_length)
    score += span_score
    
    # Replicate greedy heuristic: penalize boundary positions
    # The greedy algorithm avoids boundaries when there are alternatives
    boundary_penalty = 0.0
    
    # Check aspect span boundaries
    if aspect_span[1] == region_end:  # Aspect ends at region boundary
        boundary_penalty += 1.0
    if aspect_span[0] == region_start:  # Aspect starts at region boundary
        boundary_penalty += 1.0
    
    # Check opinion span boundaries  
    if opinion_span[0] == region_start:  # Opinion starts at region boundary
        boundary_penalty += 1.0
    if opinion_span[1] == region_end:  # Opinion ends at region boundary
        boundary_penalty += 1.0
    
    score -= boundary_penalty
    
    # Prefer spans that are well-contained within the region
    # (not touching both boundaries)
    if aspect_length < region_size and opinion_length < region_size:
        score += 0.5
    
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
    
    # Deduplicate triplets - keep highest score for each unique triplet
    triplet_dict = {}
    for triplet, score in candidate_triplets:
        triplet_key = str(triplet)
        if triplet_key not in triplet_dict or triplet_dict[triplet_key][1] < score:
            triplet_dict[triplet_key] = (triplet, score)
    
    candidate_triplets = list(triplet_dict.values())
    
    # Sort candidates by score (descending)
    candidate_triplets.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize beam with empty hypothesis
    # Each hypothesis is (triplet_list, used_triplet_strings, total_score, num_triplets)
    beam = [([], set(), 0.0, 0)]
    
    # Process each candidate
    for triplet, score in candidate_triplets:
        new_beam = []
        triplet_str = str(triplet)
        
        aspect_span, opinion_span, sentiment = triplet
        triplet_positions = set(range(aspect_span[0], aspect_span[1] + 1))
        triplet_positions.update(range(opinion_span[0], opinion_span[1] + 1))
        
        for current_triplets, used_triplet_strs, current_score, num_triplets in beam:
            # Option 1: Don't add this triplet (only keep one copy per hypothesis state)
            skip_key = (tuple(used_triplet_strs), False)
            
            # Option 2: Add this triplet if no overlap and not duplicate
            if triplet_str not in used_triplet_strs:
                # Check for position overlap with existing triplets
                has_overlap = False
                for existing_triplet in current_triplets:
                    existing_aspect, existing_opinion, _ = existing_triplet
                    existing_positions = set(range(existing_aspect[0], existing_aspect[1] + 1))
                    existing_positions.update(range(existing_opinion[0], existing_opinion[1] + 1))
                    
                    if triplet_positions & existing_positions:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    new_triplets = current_triplets + [triplet]
                    new_triplet_strs = used_triplet_strs | {triplet_str}
                    # Normalize score by number of triplets to avoid bias toward fewer triplets
                    new_score = current_score + score
                    new_beam.append((new_triplets, new_triplet_strs, new_score, num_triplets + 1))
                    # Add the skip option only if we successfully added
                    new_beam.append((current_triplets, used_triplet_strs, current_score, num_triplets))
                else:
                    # If overlap, only add skip option
                    if skip_key not in [(tuple(t[1]), False) for t in new_beam]:
                        new_beam.append((current_triplets, used_triplet_strs, current_score, num_triplets))
            else:
                # Duplicate triplet, only add skip option
                if skip_key not in [(tuple(t[1]), False) for t in new_beam]:
                    new_beam.append((current_triplets, used_triplet_strs, current_score, num_triplets))
        
        # Remove duplicate hypotheses (same triplet set)
        unique_beam = {}
        for triplets, triplet_strs, score, count in new_beam:
            key = tuple(sorted(triplet_strs))
            if key not in unique_beam or unique_beam[key][2] < score:
                unique_beam[key] = (triplets, triplet_strs, score, count)
        
        # Keep only top beam_size hypotheses, prioritize more triplets with good scores
        beam = sorted(unique_beam.values(), key=lambda x: (x[3], x[2]), reverse=True)[:beam_size]
    
    # Return the hypothesis with most triplets, breaking ties with score
    if not beam:
        return []
    
    best_hypothesis = max(beam, key=lambda x: (x[3], x[2]))
    return best_hypothesis[0]

