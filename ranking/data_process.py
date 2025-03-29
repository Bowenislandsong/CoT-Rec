import numpy as np

def pad_embeddings(embeddings, target_size=None):
    """
    Pads the input embeddings to a target size.
    
    :param embeddings: List of embeddings to pad (each embedding is a numpy array).
    :param target_size: The size to pad all embeddings to. If None, it will use the size of the longest embedding.
    :return: List of padded embeddings.
    """
    if target_size is None:
        # Find the size of the longest embedding
        target_size = max([len(embedding) for embedding in embeddings])

    padded_embeddings = []
    
    for embedding in embeddings:
        # Pad with zeros if embedding is smaller than target_size
        padded_embedding = np.pad(embedding, (0, target_size - len(embedding)), mode='constant')
        padded_embeddings.append(padded_embedding)
    
    return np.array(padded_embeddings)