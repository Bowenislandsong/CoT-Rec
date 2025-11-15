"""
Ranking evaluation metrics for recommendation systems.

This module provides standard ranking metrics including MRR, AP, MAP, and NDCG
for evaluating the quality of ranked recommendation lists.
"""

import numpy as np
from typing import List, Dict, Union


class RankingMetrics:
    """
    A collection of ranking evaluation metrics.
    
    This class provides methods for computing common ranking metrics used in
    information retrieval and recommendation systems.
    """
    
    def __init__(self):
        pass

    def mrr(self, ranked_list: List[int], ground_truth: int) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR is the reciprocal of the rank at which the first relevant item
        appears in the ranked list.
        
        Args:
            ranked_list: List of ranked item indices (from most to least relevant)
            ground_truth: Index of the correct/relevant item
        
        Returns:
            Reciprocal rank (1 / rank) for the first correct answer, or 0 if
            the correct answer is not in the ranked list
        
        Examples:
            >>> metrics = RankingMetrics()
            >>> metrics.mrr([2, 0, 1, 3], 0)
            0.5
            >>> metrics.mrr([0, 1, 2], 0)
            1.0
        """
        if not ranked_list:
            return 0.0
        
        for rank, answer_idx in enumerate(ranked_list, 1):
            if answer_idx == ground_truth:
                return 1.0 / rank
        return 0.0

    def ap(self, ranked_list: List[int], ground_truth: int) -> float:
        """
        Compute Average Precision (AP).
        
        AP measures the average of precision values at each position where
        a relevant item is found.
        
        Args:
            ranked_list: List of ranked item indices (from most to least relevant)
            ground_truth: Index of the correct/relevant item
        
        Returns:
            Average Precision score
        
        Examples:
            >>> metrics = RankingMetrics()
            >>> metrics.ap([0, 1, 2], 0)
            1.0
            >>> metrics.ap([1, 0, 2], 0)
            0.5
        """
        if not ranked_list:
            return 0.0
        
        relevant_count = 0
        ap_score = 0.0
        
        for rank, answer_idx in enumerate(ranked_list, 1):
            if answer_idx == ground_truth:
                relevant_count += 1
                ap_score += relevant_count / rank
        
        return ap_score / max(1, relevant_count)

    def map(
        self,
        ranked_lists: List[List[int]],
        ground_truths: List[int]
    ) -> float:
        """
        Compute Mean Average Precision (MAP).
        
        MAP is the mean of Average Precision scores across multiple queries.
        
        Args:
            ranked_lists: List of ranked item lists, one per query
            ground_truths: List of ground truth indices, one per query
        
        Returns:
            Mean Average Precision score
        
        Raises:
            ValueError: If ranked_lists and ground_truths have different lengths
        
        Examples:
            >>> metrics = RankingMetrics()
            >>> ranked_lists = [[0, 1, 2], [1, 0, 2]]
            >>> ground_truths = [0, 0]
            >>> metrics.map(ranked_lists, ground_truths)
            0.75
        """
        if len(ranked_lists) != len(ground_truths):
            raise ValueError(
                f"ranked_lists and ground_truths must have same length. "
                f"Got {len(ranked_lists)} and {len(ground_truths)}"
            )
        
        if not ranked_lists:
            return 0.0
        
        ap_scores = []
        for ranked_list, ground_truth in zip(ranked_lists, ground_truths):
            ap_scores.append(self.ap(ranked_list, ground_truth))
        
        return np.mean(ap_scores)

    def ndcg(
        self,
        ranked_list: List[int],
        ground_truth: int,
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.
        
        NDCG measures the quality of ranking by considering both relevance
        and position, with a logarithmic discount for lower positions.
        
        Args:
            ranked_list: List of ranked item indices (from most to least relevant)
            ground_truth: Index of the correct/relevant item
            k: Rank cut-off (e.g., top k items to consider)
        
        Returns:
            NDCG score in range [0, 1]
        
        Raises:
            ValueError: If k is less than 1
        
        Examples:
            >>> metrics = RankingMetrics()
            >>> metrics.ndcg([0, 1, 2], 0, k=3)
            1.0
            >>> metrics.ndcg([1, 0, 2], 0, k=3)
            0.63...
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        
        if not ranked_list:
            return 0.0
        
        dcg = 0.0
        for rank, answer_idx in enumerate(ranked_list[:k], 1):
            if answer_idx == ground_truth:
                # Relevance is 1 for the correct answer, 0 otherwise
                dcg += 1.0 / np.log2(rank + 1)
        
        # Ideal DCG when the correct answer is at rank 1
        ideal_dcg = 1.0 / np.log2(2)
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def evaluate(
        self,
        ranked_lists: List[List[int]],
        ground_truths: List[int],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a list of ranking results using multiple metrics.
        
        Computes MRR, AP, MAP, and NDCG@k for the given rankings.
        
        Args:
            ranked_lists: List of ranked item lists, one per query
            ground_truths: List of ground truth indices, one per query
            k: Rank cut-off for NDCG
        
        Returns:
            Dictionary containing all computed metrics:
            - MRR: Mean Reciprocal Rank
            - AP: Mean of Average Precision scores
            - MAP: Mean Average Precision
            - NDCG@k: Mean NDCG at rank k
        
        Raises:
            ValueError: If ranked_lists and ground_truths have different lengths
        
        Examples:
            >>> metrics = RankingMetrics()
            >>> ranked_lists = [[0, 1, 2], [1, 0, 2]]
            >>> ground_truths = [0, 0]
            >>> results = metrics.evaluate(ranked_lists, ground_truths, k=3)
            >>> 'MRR' in results and 'MAP' in results
            True
        """
        if len(ranked_lists) != len(ground_truths):
            raise ValueError(
                f"ranked_lists and ground_truths must have same length. "
                f"Got {len(ranked_lists)} and {len(ground_truths)}"
            )
        
        if not ranked_lists:
            return {
                "MRR": 0.0,
                "AP": 0.0,
                "MAP": 0.0,
                "NDCG@k": 0.0
            }
        
        mrr_scores = [
            self.mrr(ranked_list, ground_truth)
            for ranked_list, ground_truth in zip(ranked_lists, ground_truths)
        ]
        ap_scores = [
            self.ap(ranked_list, ground_truth)
            for ranked_list, ground_truth in zip(ranked_lists, ground_truths)
        ]
        ndcg_scores = [
            self.ndcg(ranked_list, ground_truth, k)
            for ranked_list, ground_truth in zip(ranked_lists, ground_truths)
        ]

        metrics = {
            "MRR": np.mean(mrr_scores),
            "AP": np.mean(ap_scores),
            "MAP": self.map(ranked_lists, ground_truths),
            "NDCG@k": np.mean(ndcg_scores)
        }
        
        return metrics
