"""
Unit tests for ranking models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestXGBoostRanker:
    """Test suite for XGBoostRanker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import here to avoid import errors if xgboost is not installed
        try:
            from ranking.xgboost_ranker import XGBoostRanker
            self.XGBoostRanker = XGBoostRanker
            self.xgboost_available = True
        except ImportError:
            self.xgboost_available = False
    
    def test_initialization(self):
        """Test XGBoostRanker initialization."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        assert ranker.model is None
        assert ranker.max_len is None
        assert 'objective' in ranker.params
        assert 'eval_metric' in ranker.params
    
    def test_prepare_data_basic(self):
        """Test basic data preparation."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        input_tokens = [np.random.rand(5), np.random.rand(5)]
        output_tokens = [
            [np.random.rand(4), np.random.rand(4)],
            [np.random.rand(4), np.random.rand(4)]
        ]
        scores = [[1, 0], [0, 1]]
        
        X, y, group = ranker.prepare_data(input_tokens, output_tokens, scores)
        
        assert X.shape[0] == 4  # 2 inputs * 2 candidates each
        assert y.shape[0] == 4
        assert len(group) == 2
        assert group[0] == 2
        assert group[1] == 2
    
    def test_prepare_data_sets_max_len(self):
        """Test that prepare_data sets max_len correctly."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        input_tokens = [np.random.rand(5)]
        output_tokens = [[np.random.rand(3)]]
        scores = [[1]]
        
        X, y, group = ranker.prepare_data(input_tokens, output_tokens, scores)
        
        # max_len should be 5 + 3 = 8
        assert ranker.max_len == 8
        assert X.shape[1] == 8
    
    def test_train_without_validation(self):
        """Test training without validation data."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        # Create simple training data
        input_tokens = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        output_tokens = [
            [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            [np.array([9.0, 10.0]), np.array([11.0, 12.0])]
        ]
        scores = [[1, 0], [0, 1]]
        
        model = ranker.train(input_tokens, output_tokens, scores)
        
        assert model is not None
        assert ranker.model is not None
    
    def test_predict_without_training_raises_error(self):
        """Test that predict raises error if model not trained."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        input_tokens = [np.array([1.0, 2.0])]
        output_tokens = [[np.array([3.0, 4.0])]]
        
        with pytest.raises(ValueError, match="Model not trained"):
            ranker.predict(input_tokens, output_tokens)
    
    def test_train_and_predict(self):
        """Test full training and prediction pipeline."""
        if not self.xgboost_available:
            pytest.skip("XGBoost not available")
        
        ranker = self.XGBoostRanker(use_gpu=False)
        
        # Create training data
        np.random.seed(42)
        input_tokens = [np.random.rand(5) for _ in range(3)]
        output_tokens = [
            [np.random.rand(4), np.random.rand(4)]
            for _ in range(3)
        ]
        scores = [[1, 0], [0, 1], [1, 0]]
        
        # Train
        ranker.train(input_tokens, output_tokens, scores)
        
        # Predict
        predictions = ranker.predict(input_tokens, output_tokens)
        
        assert len(predictions) == 3
        assert all(len(pred) == 2 for pred in predictions)


class TestRankingHead:
    """Test suite for RankingHead and Ranker classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import here to avoid import errors if torch is not installed
        try:
            import torch
            from ranking.ranking_head_ranker import RankingHead, Ranker
            self.torch = torch
            self.RankingHead = RankingHead
            self.Ranker = Ranker
            self.torch_available = True
        except ImportError:
            self.torch_available = False
    
    def test_ranking_head_initialization(self):
        """Test RankingHead model initialization."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        model = self.RankingHead(input_dim=64)
        
        assert model is not None
        # Check that model has the expected structure
        assert hasattr(model, 'fc')
    
    def test_ranking_head_forward(self):
        """Test RankingHead forward pass."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        model = self.RankingHead(input_dim=64)
        x = self.torch.randn(10, 64)
        
        output = model(x)
        
        assert output.shape == (10,)
    
    def test_ranker_initialization(self):
        """Test Ranker initialization."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        ranker = self.Ranker(max_len=512)
        
        assert ranker.max_len == 512
        assert ranker.model is None
        assert ranker.device is not None
    
    def test_ranker_prepare_data(self):
        """Test Ranker data preparation."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        ranker = self.Ranker(max_len=20)
        
        input_tokens = [np.random.rand(5), np.random.rand(5)]
        output_tokens = [
            [np.random.rand(4), np.random.rand(4)],
            [np.random.rand(4), np.random.rand(4)]
        ]
        scores = [[1, 0], [0, 1]]
        
        X, y = ranker.prepare_data(input_tokens, output_tokens, scores)
        
        assert X.shape[0] == 4  # 2 inputs * 2 candidates
        assert X.shape[1] == 20  # max_len
        assert y.shape[0] == 4
    
    def test_ranker_train_basic(self):
        """Test basic training functionality."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        ranker = self.Ranker(max_len=20)
        
        # Create simple training data
        np.random.seed(42)
        input_tokens = [np.random.rand(5) for _ in range(2)]
        output_tokens = [
            [np.random.rand(4), np.random.rand(4)]
            for _ in range(2)
        ]
        scores = [[1, 0], [0, 1]]
        
        # Train for just 1 epoch to test functionality
        ranker.train(
            input_tokens,
            output_tokens,
            scores,
            epochs=1,
            batch_size=2,
            lr=1e-3
        )
        
        assert ranker.model is not None
    
    def test_ranker_predict(self):
        """Test prediction functionality."""
        if not self.torch_available:
            pytest.skip("PyTorch not available")
        
        ranker = self.Ranker(max_len=20)
        
        # Create and train on simple data
        np.random.seed(42)
        input_tokens = [np.random.rand(5) for _ in range(2)]
        output_tokens = [
            [np.random.rand(4), np.random.rand(4)]
            for _ in range(2)
        ]
        scores = [[1, 0], [0, 1]]
        
        ranker.train(
            input_tokens,
            output_tokens,
            scores,
            epochs=1,
            batch_size=2,
            lr=1e-3
        )
        
        # Predict
        predictions = ranker.predict(input_tokens, output_tokens)
        
        assert len(predictions) == 4  # Flattened predictions
        assert all(isinstance(p, (float, np.floating)) for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
