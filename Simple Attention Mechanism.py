import numpy as np

def simple_attention(query, keys, values):
    """
    Simple dot-product attention mechanism.
    
    Args:
    query: The query vector (shape: [query_dim]).
    keys: The key vectors (shape: [num_keys, key_dim]).
    values: The value vectors (shape: [num_keys, value_dim]).
    
    Returns:
    Attention-weighted sum of the values.
    """
    # Compute attention scores
    scores = np.dot(keys, query)
    
    # Convert scores to probabilities using softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Compute the attention-weighted sum of values
    weighted_sum = np.dot(attention_weights, values)
    
    return weighted_sum

# Example usage
query = np.array([0.5, 0.2])  # Example query vector
keys = np.array([[0.1, 0.3],
                 [0.2, 0.5],
                 [0.3, 0.7]])  # Example key vectors
values = np.array([[1, 2],
                   [2, 4],
                   [3, 6]])  # Example value vectors

result = simple_attention(query, keys, values)
print("Attention-weighted sum:", result)
