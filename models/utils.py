import torch
import torch.nn.functional as F

def _ensure_valid_dims(*tensors):
    """Helper to check tensor dimensions."""
    for t in tensors:
        if t.dim() not in [2, 3]:
             raise ValueError(f"Expected 2D or 3D tensor, got shape {t.shape}")

def cosine_similarity(keys: torch.Tensor, memory: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Calculates batched cosine similarity with broadcasting for shared memory.

    Args:
        keys: Tensor (batch_size, num_keys, key_size)
        memory: Tensor (batch_size, mem_slots, key_size) OR (mem_slots, key_size) if shared.
        epsilon: Small value for numerical stability.

    Returns:
        Tensor (batch_size, num_keys, mem_slots) of similarities.
    """
    _ensure_valid_dims(keys)
    if not (memory.dim() == 2 or (memory.dim() == 3 and memory.size(0) == keys.size(0))):
         raise ValueError(f"Memory shape {memory.shape} incompatible with keys shape {keys.shape}")

    # Normalize keys and memory vectors
    keys_norm = torch.norm(keys, p=2, dim=-1, keepdim=True)
    memory_norm = torch.norm(memory, p=2, dim=-1, keepdim=True)

    # Add epsilon to avoid division by zero
    keys_normalized = keys / (keys_norm + epsilon)
    memory_normalized = memory / (memory_norm + epsilon)

    # Prepare memory for batch matrix multiplication
    if memory_normalized.dim() == 2: # Shared memory: (N, M) -> (1, N, M) -> (1, M, N)
        memory_transposed = memory_normalized.t().unsqueeze(0)
    else: # Batched memory: (B, N, M) -> (B, M, N)
        memory_transposed = memory_normalized.transpose(1, 2)

    # Perform batch matrix multiplication: (B, K, S) @ (B or 1, S, N) -> (B, K, N)
    similarity = torch.bmm(keys_normalized, memory_transposed)

    # Clamp similarity to avoid potential numerical issues outside [-1, 1]
    similarity = torch.clamp(similarity, -1.0, 1.0)

    return similarity

def outer_product(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Calculates batched outer product.

    Args:
        vec1: Tensor (batch_size, N)
        vec2: Tensor (batch_size, M)

    Returns:
        Tensor (batch_size, N, M) where out[b, i, j] = vec1[b, i] * vec2[b, j]
    """
    if not (vec1.dim() == 2 and vec2.dim() == 2):
         raise ValueError(f"Inputs must be 2D tensors (batch_size, dim). Got {vec1.shape} and {vec2.shape}")
    if vec1.size(0) != vec2.size(0):
        raise ValueError(f"Batch sizes must match. Got {vec1.size(0)} and {vec2.size(0)}")

    # vec1: (B, N) -> (B, N, 1)
    # vec2: (B, M) -> (B, 1, M)
    # Result: (B, N, M) via broadcasting bmm
    return torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1))


def sparse_softmax_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Applies softmax only to the top k elements, setting others to zero.

    Args:
        logits: Tensor of shape (batch_size, ..., num_elements)
        k: The number of top elements to consider.

    Returns:
        Tensor with the same shape as logits, where non-top-k elements have zero probability.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    num_elements = logits.size(-1)
    k = min(k, num_elements) # Ensure k is not larger than the number of elements

    if k == num_elements: # If k covers all elements, just use standard softmax
        return F.softmax(logits, dim=-1)

    # Find the top k values and their indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create a mask, initializing with a very small number that leads to zero after softmax
    mask_value = torch.finfo(logits.dtype).min
    masked_logits = torch.full_like(logits, mask_value)

    # Place the top k values back into the masked logits tensor
    # scatter_(dim, index, src) -> self[index[i][j]][j] = src[i][j] (simplified)
    # Need correct dimension index
    masked_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

    # Apply softmax - elements not in top-k will have near-zero probability
    sparse_weights = F.softmax(masked_logits, dim=-1)

    # Optional: Explicitly zero out tiny values if needed, though softmax handles this well
    # threshold = 1e-9
    # sparse_weights[sparse_weights < threshold] = 0.0

    return sparse_weights

def weighted_softmax(weights: torch.Tensor, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes softmax weighted by external weights.

    Args:
        weights: Tensor of weights (same shape as logits or broadcastable).
        logits: Input logits.
        dim: Dimension along which to compute softmax.

    Returns:
        Weighted softmax probabilities.
    """
    # Ensure weights are positive (e.g., using exp or softplus if they aren't already)
    # Assuming weights are probabilities or positive scores here.
    weighted_logits = weights * logits
    return F.softmax(weighted_logits, dim=dim)

