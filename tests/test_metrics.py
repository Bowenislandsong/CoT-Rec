"""
Unit tests for ranking metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking.metrics import RankingMetrics


class TestRankingMetrics:
    """Test suite for RankingMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = RankingMetrics()
    
    # Tests for MRR
    def test_mrr_first_position(self):
        """Test MRR when correct answer is at first position."""
        ranked_list = [0, 1, 2, 3]
        ground_truth = 0
        
        result = self.metrics.mrr(ranked_list, ground_truth)
        
        assert result == 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when correct answer is at second position."""
        ranked_list = [1, 0, 2, 3]
        ground_truth = 0
        
        result = self.metrics.mrr(ranked_list, ground_truth)
        
        assert result == 0.5
    
    def test_mrr_not_found(self):
        """Test MRR when correct answer is not in the list."""
        ranked_list = [1, 2, 3]
        ground_truth = 0
        
        result = self.metrics.mrr(ranked_list, ground_truth)
        
        assert result == 0.0
    
    def test_mrr_empty_list(self):
        """Test MRR with empty ranked list."""
        ranked_list = []
        ground_truth = 0
        
        result = self.metrics.mrr(ranked_list, ground_truth)
        
        assert result == 0.0
    
    # Tests for AP
    def test_ap_perfect_ranking(self):
        """Test AP with perfect ranking."""
        ranked_list = [0, 1, 2]
        ground_truth = 0
        
        result = self.metrics.ap(ranked_list, ground_truth)
        
        assert result == 1.0
    
    def test_ap_second_position(self):
        """Test AP when correct answer is at second position."""
        ranked_list = [1, 0, 2]
        ground_truth = 0
        
        result = self.metrics.ap(ranked_list, ground_truth)
        
        assert result == 0.5
    
    def test_ap_not_found(self):
        """Test AP when correct answer is not in the list."""
        ranked_list = [1, 2, 3]
        ground_truth = 0
        
        result = self.metrics.ap(ranked_list, ground_truth)
        
        assert result == 0.0
    
    def test_ap_empty_list(self):
        """Test AP with empty ranked list."""
        ranked_list = []
        ground_truth = 0
        
        result = self.metrics.ap(ranked_list, ground_truth)
        
        assert result == 0.0
    
    # Tests for MAP
    def test_map_multiple_queries(self):
        """Test MAP with multiple queries."""
        ranked_lists = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
        ground_truths = [0, 0, 0]
        
        result = self.metrics.map(ranked_lists, ground_truths)
        
        expected = (1.0 + 0.5 + 1.0) / 3
        assert abs(result - expected) < 1e-6
    
    def test_map_single_query(self):
        """Test MAP with single query."""
        ranked_lists = [[0, 1, 2]]
        ground_truths = [0]
        
        result = self.metrics.map(ranked_lists, ground_truths)
        
        assert result == 1.0
    
    def test_map_mismatched_lengths_raises_error(self):
        """Test MAP raises error when input lengths don't match."""
        ranked_lists = [[0, 1, 2], [1, 0, 2]]
        ground_truths = [0]
        
        with pytest.raises(ValueError, match="must have same length"):
            self.metrics.map(ranked_lists, ground_truths)
    
    def test_map_empty_lists(self):
        """Test MAP with empty lists."""
        ranked_lists = []
        ground_truths = []
        
        result = self.metrics.map(ranked_lists, ground_truths)
        
        assert result == 0.0
    
    # Tests for NDCG
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        ranked_list = [0, 1, 2]
        ground_truth = 0
        
        result = self.metrics.ndcg(ranked_list, ground_truth, k=3)
        
        assert result == 1.0
    
    def test_ndcg_second_position(self):
        """Test NDCG when correct answer is at second position."""
        ranked_list = [1, 0, 2]
        ground_truth = 0
        k = 3
        
        result = self.metrics.ndcg(ranked_list, ground_truth, k=k)
        
        # DCG = 1/log2(3) = 0.63...
        # IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
        assert abs(result - expected) < 1e-6
    
    def test_ndcg_not_in_top_k(self):
        """Test NDCG when correct answer is not in top k."""
        ranked_list = [1, 2, 3, 0]
        ground_truth = 0
        k = 2
        
        result = self.metrics.ndcg(ranked_list, ground_truth, k=k)
        
        assert result == 0.0
    
    def test_ndcg_invalid_k_raises_error(self):
        """Test NDCG raises error for invalid k."""
        ranked_list = [0, 1, 2]
        ground_truth = 0
        
        with pytest.raises(ValueError, match="k must be at least 1"):
            self.metrics.ndcg(ranked_list, ground_truth, k=0)
    
    def test_ndcg_empty_list(self):
        """Test NDCG with empty ranked list."""
        ranked_list = []
        ground_truth = 0
        
        result = self.metrics.ndcg(ranked_list, ground_truth, k=10)
        
        assert result == 0.0
    
    # Tests for evaluate
    def test_evaluate_returns_all_metrics(self):
        """Test that evaluate returns all expected metrics."""
        ranked_lists = [[0, 1, 2], [1, 0, 2]]
        ground_truths = [0, 0]
        
        result = self.metrics.evaluate(ranked_lists, ground_truths, k=3)
        
        assert "MRR" in result
        assert "AP" in result
        assert "MAP" in result
        assert "NDCG@k" in result
    
    def test_evaluate_correct_values(self):
        """Test that evaluate computes correct values."""
        ranked_lists = [[0, 1, 2]]
        ground_truths = [0]
        
        result = self.metrics.evaluate(ranked_lists, ground_truths, k=3)
        
        assert result["MRR"] == 1.0
        assert result["AP"] == 1.0
        assert result["MAP"] == 1.0
        assert result["NDCG@k"] == 1.0
    
    def test_evaluate_mismatched_lengths_raises_error(self):
        """Test evaluate raises error when input lengths don't match."""
        ranked_lists = [[0, 1, 2], [1, 0, 2]]
        ground_truths = [0]
        
        with pytest.raises(ValueError, match="must have same length"):
            self.metrics.evaluate(ranked_lists, ground_truths)
    
    def test_evaluate_empty_lists(self):
        """Test evaluate with empty lists."""
        ranked_lists = []
        ground_truths = []
        
        result = self.metrics.evaluate(ranked_lists, ground_truths)
        
        assert result["MRR"] == 0.0
        assert result["AP"] == 0.0
        assert result["MAP"] == 0.0
        assert result["NDCG@k"] == 0.0
    
    def test_evaluate_multiple_queries(self):
        """Test evaluate with multiple queries."""
        ranked_lists = [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 1, 0, 3]
        ]
        ground_truths = [0, 0, 0]
        
        result = self.metrics.evaluate(ranked_lists, ground_truths, k=4)
        
        # Check that all metrics are between 0 and 1
        assert 0 <= result["MRR"] <= 1
        assert 0 <= result["AP"] <= 1
        assert 0 <= result["MAP"] <= 1
        assert 0 <= result["NDCG@k"] <= 1
        
        # MRR should be (1.0 + 0.5 + 1/3) / 3
        expected_mrr = (1.0 + 0.5 + 1.0/3) / 3
        assert abs(result["MRR"] - expected_mrr) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
