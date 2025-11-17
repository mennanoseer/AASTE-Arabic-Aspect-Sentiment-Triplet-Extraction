"""
Maximum Weight Independent Set (MWIS) Solver for Interval Graphs.

This module implements an optimal dynamic programming algorithm for selecting
the highest-scoring set of non-overlapping spans. This is a provably optimal
solution to the weighted interval scheduling problem, which is equivalent to
finding the Maximum Weight Independent Set on an interval graph.

The algorithm runs in O(N log N) time, where N is the number of candidate spans.

References:
    - Weighted Interval Scheduling: Kleinberg & Tardos, Algorithm Design
    - MWIS on Interval Graphs: Golumbic, Algorithmic Graph Theory
"""

import torch
from typing import List, Tuple, Dict
import bisect


class MWISIntervalSolver:
    """
    Solver for Maximum Weight Independent Set problem on interval graphs.
    
    This class provides optimal span selection for span-based NLP tasks,
    replacing greedy heuristics with provably optimal dynamic programming.
    """
    
    def __init__(self):
        """Initialize the MWIS solver."""
        pass
    
    def solve(self, spans: List[Tuple[int, int]], weights: List[float]) -> List[int]:
        """
        Find the maximum weight independent set of non-overlapping spans.
        
        Args:
            spans: List of span tuples (start_idx, end_idx) where indices are inclusive
            weights: List of weights/scores corresponding to each span
            
        Returns:
            List of indices of selected spans (indices into the input spans list)
            
        Example:
            >>> solver = MWISIntervalSolver()
            >>> spans = [(0, 1), (0, 2), (1, 2), (2, 3)]
            >>> weights = [0.5, 0.8, 0.6, 0.9]
            >>> selected = solver.solve(spans, weights)
            >>> print([spans[i] for i in selected])
            [(0, 2), (2, 3)]  # Non-overlapping spans with max total weight
        """
        if not spans:
            return []
        
        n = len(spans)
        
        # Step 1: Create indexed spans and sort by end point
        indexed_spans = [(spans[i], weights[i], i) for i in range(n)]
        indexed_spans.sort(key=lambda x: x[0][1])  # Sort by end index
        
        # Step 2: Compute p(k) for each span k
        # p(k) is the rightmost span that doesn't overlap with span k
        p = self._compute_compatible_predecessors(indexed_spans)
        
        # Step 3: Dynamic Programming
        # W[k] = maximum weight achievable using first k spans
        W = [0.0] * (n + 1)
        
        for k in range(1, n + 1):
            span, weight, _ = indexed_spans[k - 1]
            # Choice 1: Don't include span k
            exclude_weight = W[k - 1]
            # Choice 2: Include span k
            include_weight = weight + W[p[k - 1]]
            # Take the better choice
            W[k] = max(exclude_weight, include_weight)
        
        # Step 4: Backtrack to find which spans were selected
        selected_indices = self._backtrack(indexed_spans, W, p)
        
        return selected_indices
    
    def _compute_compatible_predecessors(
        self, 
        indexed_spans: List[Tuple[Tuple[int, int], float, int]]
    ) -> List[int]:
        """
        Compute p(k) for each span: the index of the rightmost compatible predecessor.
        
        A span j is compatible with span k if it doesn't overlap with k.
        Since spans are sorted by end point, we look for the rightmost span
        whose end point is before span k's start point.
        
        Args:
            indexed_spans: List of (span, weight, original_index) sorted by end point
            
        Returns:
            List where p[k] is the index such that span p[k] is compatible with span k
        """
        n = len(indexed_spans)
        p = []
        end_points = []
        
        for k in range(n):
            span, _, _ = indexed_spans[k]
            start_k = span[0]
            
            # Find the rightmost span j where end_j < start_k
            # Using binary search for O(log n) lookup
            idx = bisect.bisect_right(end_points, start_k - 1) - 1
            p.append(idx + 1)  # +1 because W is 1-indexed (W[0] = 0)
            
            end_points.append(span[1])
        
        return p
    
    def _backtrack(
        self,
        indexed_spans: List[Tuple[Tuple[int, int], float, int]],
        W: List[float],
        p: List[int]
    ) -> List[int]:
        """
        Backtrack through the DP table to find which spans were selected.
        
        Args:
            indexed_spans: List of (span, weight, original_index) sorted by end point
            W: DP table where W[k] is the max weight for first k spans
            p: Predecessor array
            
        Returns:
            List of original indices of selected spans
        """
        selected = []
        k = len(indexed_spans)
        
        while k > 0:
            span, weight, original_idx = indexed_spans[k - 1]
            # Check if span k was included in the optimal solution
            if weight + W[p[k - 1]] >= W[k - 1]:
                # Span k was included
                selected.append(original_idx)
                k = p[k - 1]  # Jump to the compatible predecessor
            else:
                # Span k was not included
                k -= 1
        
        # Return in original order (sorted by start position)
        selected.reverse()
        return selected
    
    def solve_with_scores_tensor(
        self, 
        span_scores: torch.Tensor, 
        min_score: float = 0.0
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Extract optimal non-overlapping spans from a score tensor.
        
        This is a convenience method that takes a 2D tensor of span scores
        (where [i, j] is the score for span from i to j) and returns the
        optimal set of non-overlapping spans.
        
        Args:
            span_scores: 2D tensor of shape (seq_len, seq_len) with span scores
            min_score: Minimum score threshold for considering a span
            
        Returns:
            Tuple of (selected_spans, selected_scores) where:
                - selected_spans: List of (start, end) tuples
                - selected_scores: List of corresponding scores
        """
        # Extract all candidate spans above threshold
        seq_len = span_scores.shape[0]
        candidate_spans = []
        candidate_weights = []
        
        for i in range(seq_len):
            for j in range(i, seq_len):
                score = span_scores[i, j].item()
                if score > min_score:
                    candidate_spans.append((i, j))
                    candidate_weights.append(score)
        
        # Solve MWIS
        if not candidate_spans:
            return [], []
        
        selected_indices = self.solve(candidate_spans, candidate_weights)
        
        # Extract selected spans and scores
        selected_spans = [candidate_spans[idx] for idx in selected_indices]
        selected_scores = [candidate_weights[idx] for idx in selected_indices]
        
        return selected_spans, selected_scores


def extract_optimal_spans_from_mask(
    span_mask: torch.Tensor,
    span_scores: torch.Tensor,
    offset: int = 0,
    min_score: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Extract optimal non-overlapping spans from a boolean mask and score tensor.
    
    This function is designed to replace greedy span extraction in the
    original inference code.
    
    Args:
        span_mask: Boolean tensor indicating potential span positions
        span_scores: Float tensor with scores for each span position
        offset: Offset to add to span indices (for sub-sequence extraction)
        min_score: Minimum score threshold
        
    Returns:
        List of selected span tuples (start, end) with offset applied
    """
    solver = MWISIntervalSolver()
    
    # Get candidate spans from mask
    candidate_indices = span_mask.nonzero()
    if len(candidate_indices) == 0:
        return []
    
    # For 1D masks (sequence of boolean values)
    if candidate_indices.dim() == 1 or candidate_indices.shape[1] == 1:
        indices = candidate_indices.squeeze().tolist()
        if not isinstance(indices, list):
            indices = [indices]
        
        # Create spans of single positions
        spans = [(idx + offset, idx + offset) for idx in indices]
        weights = [span_scores[idx].item() if span_scores.dim() == 1 else span_scores[idx, idx].item() 
                   for idx in indices]
    else:
        # For 2D masks (span grid)
        spans = [(idx[0].item() + offset, idx[1].item() + offset) 
                 for idx in candidate_indices]
        weights = [span_scores[idx[0], idx[1]].item() for idx in candidate_indices]
    
    # Filter by minimum score
    filtered_spans = []
    filtered_weights = []
    for span, weight in zip(spans, weights):
        if weight >= min_score:
            filtered_spans.append(span)
            filtered_weights.append(weight)
    
    if not filtered_spans:
        return []
    
    # Solve MWIS
    selected_indices = solver.solve(filtered_spans, filtered_weights)
    selected_spans = [filtered_spans[idx] for idx in selected_indices]
    
    return selected_spans
