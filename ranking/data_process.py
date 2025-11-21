"""
Data processing utilities for embedding manipulation and preparation.

This module provides functions for preparing embeddings for ranking models,
including padding operations to handle variable-length sequences.
"""

import numpy as np
from typing import List, Optional, Union


def pad_embeddings(
    embeddings: List[np.ndarray],
    target_size: Optional[int] = None
) -> np.ndarray:
    """
    Pads the input embeddings to a uniform target size.
    
    This function ensures all embeddings have the same dimensionality by
    padding shorter embeddings with zeros. This is essential for batch
    processing in neural networks and tree-based models.
    
    Args:
        embeddings: List of embeddings to pad. Each embedding should be a
                   1D numpy array or array-like object.
        target_size: The size to pad all embeddings to. If None, the size
                    of the longest embedding will be used automatically.
    
    Returns:
        A 2D numpy array of shape (n_embeddings, target_size) containing
        the padded embeddings.
    
    Raises:
        ValueError: If embeddings list is empty.
        ValueError: If target_size is specified and is smaller than the
                   longest embedding.
        TypeError: If embeddings is not a list or if individual embeddings
                  cannot be converted to numpy arrays.
    
    Examples:
        >>> emb1 = np.array([1.0, 2.0, 3.0])
        >>> emb2 = np.array([4.0, 5.0])
        >>> padded = pad_embeddings([emb1, emb2])
        >>> print(padded)
        [[1. 2. 3.]
         [4. 5. 0.]]
        
        >>> padded = pad_embeddings([emb1, emb2], target_size=5)
        >>> print(padded.shape)
        (2, 5)
    """
    if not isinstance(embeddings, list):
        raise TypeError(f"embeddings must be a list, got {type(embeddings)}")
    
    if len(embeddings) == 0:
        raise ValueError("embeddings list cannot be empty")
    
    # Convert embeddings to numpy arrays if they aren't already
    try:
        embeddings = [np.asarray(emb) for emb in embeddings]
    except Exception as e:
        raise TypeError(f"Failed to convert embeddings to numpy arrays: {e}")
    
    # Validate that all embeddings are 1D
    for i, emb in enumerate(embeddings):
        if emb.ndim != 1:
            raise ValueError(
                f"All embeddings must be 1D arrays. "
                f"Embedding at index {i} has shape {emb.shape}"
            )
    
    # Determine target size
    max_size = max(len(embedding) for embedding in embeddings)
    
    if target_size is None:
        target_size = max_size
    elif target_size < max_size:
        raise ValueError(
            f"target_size ({target_size}) must be at least as large as "
            f"the longest embedding ({max_size})"
        )

    padded_embeddings = []
    
    for embedding in embeddings:
        # Pad with zeros if embedding is smaller than target_size
        pad_width = target_size - len(embedding)
        padded_embedding = np.pad(
            embedding,
            (0, pad_width),
            mode='constant',
            constant_values=0
        )
        padded_embeddings.append(padded_embedding)
    
    return np.array(padded_embeddings)