"""
Unit tests for data processing utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking.data_process import pad_embeddings


class TestPadEmbeddings:
    """Test suite for the pad_embeddings function."""
    
    def test_pad_basic(self):
        """Test basic padding functionality."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0])
        
        result = pad_embeddings([emb1, emb2])
        
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[1], [4.0, 5.0, 0.0])
    
    def test_pad_with_target_size(self):
        """Test padding with explicit target size."""
        emb1 = np.array([1.0, 2.0])
        emb2 = np.array([3.0])
        
        result = pad_embeddings([emb1, emb2], target_size=5)
        
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[1], [3.0, 0.0, 0.0, 0.0, 0.0])
    
    def test_pad_single_embedding(self):
        """Test padding with a single embedding."""
        emb = np.array([1.0, 2.0, 3.0])
        
        result = pad_embeddings([emb])
        
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result[0], emb)
    
    def test_pad_equal_size_embeddings(self):
        """Test padding when all embeddings are already equal size."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0, 6.0])
        
        result = pad_embeddings([emb1, emb2])
        
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], emb1)
        np.testing.assert_array_equal(result[1], emb2)
    
    def test_pad_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="embeddings list cannot be empty"):
            pad_embeddings([])
    
    def test_pad_invalid_target_size_raises_error(self):
        """Test that invalid target_size raises ValueError."""
        emb = np.array([1.0, 2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError, match="target_size.*must be at least"):
            pad_embeddings([emb], target_size=2)
    
    def test_pad_non_list_input_raises_error(self):
        """Test that non-list input raises TypeError."""
        with pytest.raises(TypeError, match="embeddings must be a list"):
            pad_embeddings(np.array([1.0, 2.0]))
    
    def test_pad_2d_embedding_raises_error(self):
        """Test that 2D embeddings raise ValueError."""
        emb = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="All embeddings must be 1D"):
            pad_embeddings([emb])
    
    def test_pad_preserves_dtype(self):
        """Test that padding preserves the data type."""
        emb1 = np.array([1, 2, 3], dtype=np.int32)
        emb2 = np.array([4, 5], dtype=np.int32)
        
        result = pad_embeddings([emb1, emb2])
        
        # Result should be an array
        assert isinstance(result, np.ndarray)
    
    def test_pad_list_input(self):
        """Test that list inputs are converted to numpy arrays."""
        emb1 = [1.0, 2.0, 3.0]
        emb2 = [4.0, 5.0]
        
        result = pad_embeddings([emb1, emb2])
        
        assert result.shape == (2, 3)
        assert isinstance(result, np.ndarray)
    
    def test_pad_large_embeddings(self):
        """Test padding with larger embeddings."""
        emb1 = np.random.rand(100)
        emb2 = np.random.rand(50)
        emb3 = np.random.rand(75)
        
        result = pad_embeddings([emb1, emb2, emb3])
        
        assert result.shape == (3, 100)
        np.testing.assert_array_equal(result[0], emb1)
        np.testing.assert_array_equal(result[1][:50], emb2)
        np.testing.assert_array_equal(result[1][50:], np.zeros(50))
    
    def test_pad_zero_length_embedding(self):
        """Test behavior with zero-length embedding."""
        emb1 = np.array([1.0, 2.0])
        emb2 = np.array([])
        
        result = pad_embeddings([emb1, emb2])
        
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], emb1)
        np.testing.assert_array_equal(result[1], [0.0, 0.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
