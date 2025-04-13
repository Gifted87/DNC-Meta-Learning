import torch
from torch import nn
import torch.nn.functional as F
from .utils import cosine_similarity, outer_product, sparse_softmax_topk

class MemoryReadHead(nn.Module):
    """Reads from memory using content addressing and potentially location/temporal addressing."""
    def __init__(self,
                 memory_slots: int,
                 memory_vector_size: int,
                 hidden_size: int,
                 num_read_heads: int, # Needed for temporal link calculations if implemented fully
                 k_sparse_read: int):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_vector_size = memory_vector_size
        self.hidden_size = hidden_size
        self.num_read_heads = num_read_heads
        self.k_sparse_read = k_sparse_read # For sparse content reading

        # Parameters for content-based addressing
        self.key_proj = nn.Linear(hidden_size, memory_vector_size)
        self.strength_proj = nn.Linear(hidden_size, 1)

        # Parameters for location-based addressing (using previous read weights & link matrix)
        # Gating parameter to interpolate between content and location addressing
        self.interpolation_gate_proj = nn.Linear(hidden_size, 1) # For read mode (b, f, c)

        # Projection for forward/backward temporal weights (if using links)
        # self.temporal_proj = nn.Linear(hidden_size, num_read_heads * 3) # g^f, g^b - complex DNC feature

    def forward(self,
                h: torch.Tensor,
                memory: torch.Tensor,
                prev_read_weights: torch.Tensor, # Shape: (B, num_read_heads, N)
                link_matrix: torch.Tensor # Shape: (B, N, N) - Needed for temporal reads
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Controller hidden state (batch_size, hidden_size)
            memory: Memory tensor (batch_size, memory_slots, memory_vector_size)
            prev_read_weights: Previous read weights for *all* heads (B, num_heads, N)
            link_matrix: Temporal link matrix (B, N, N)

        Returns:
            read_vector: (batch_size, memory_vector_size)
            current_read_weights: (batch_size, memory_slots) - Weights for *this* head
        """
        batch_size = h.size(0)
        device = h.device

        # --- 1. Content Addressing ---
        read_key = self.key_proj(h) # (B, M)
        read_strength = F.softplus(self.strength_proj(h)) # (B, 1) Ensure positivity
        # Calculate similarity: (B, 1, M) @ (B, M, N) -> (B, 1, N) -> (B, N)
        similarity = cosine_similarity(read_key.unsqueeze(1), memory).squeeze(1)
        content_logits = read_strength * similarity # (B, N)
        # Apply sparse softmax for content weights
        content_weights = sparse_softmax_topk(content_logits, self.k_sparse_read) # (B, N)

        # --- 2. Location Addressing (Simplified Temporal - based on prev read) ---
        # The full DNC paper uses complex forward/backward weights from the link matrix.
        # Here, we use a simpler interpolation between content and previous read weights.
        # Note: This requires passing the *specific* previous read weight for *this* head.
        # How to know which head this is? Assume index is passed or managed externally.
        # **Simplification:** Let's use average prev reads or skip temporal for now.
        # **Decision:** For simplicity in this version, skip temporal reads. Focus on content.
        # If implementing temporal:
        #   - Need to know the index of *this* head.
        #   - Calculate forward weights: w_f = bmm(link_matrix, prev_read_weights_this_head.unsqueeze(2)).squeeze(2)
        #   - Calculate backward weights: w_b = bmm(link_matrix.transpose(1,2), prev_read_weights_this_head.unsqueeze(2)).squeeze(2)
        #   - Interpolate using gates derived from controller state h.

        # --- 3. Final Read Weights (Simplified: only content-based) ---
        # Gate between content and location (skipped location, so gate is irrelevant now)
        # interpolation_gate = torch.sigmoid(self.interpolation_gate_proj(h)) # (B, 1)
        # Final weights would be interpolation_gate * location_weights + (1 - interpolation_gate) * content_weights

        current_read_weights = content_weights # Using only content weights

        # --- 4. Read from Memory ---
        # read_weights: (B, N) -> (B, 1, N)
        # memory: (B, N, M)
        # result: (B, 1, M) -> (B, M)
        read_vector = torch.bmm(current_read_weights.unsqueeze(1), memory).squeeze(1)

        return read_vector, current_read_weights


class MemoryWriteHead(nn.Module):
    """Writes to memory using content addressing and allocation."""
    def __init__(self,
                 memory_slots: int,
                 memory_vector_size: int,
                 hidden_size: int):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_vector_size = memory_vector_size
        self.hidden_size = hidden_size

        # Parameters for content-based addressing (write key)
        self.key_proj = nn.Linear(hidden_size, memory_vector_size)
        self.strength_proj = nn.Linear(hidden_size, 1)

        # Parameters for memory modification
        self.erase_vec_proj = nn.Linear(hidden_size, memory_vector_size)
        self.add_vec_proj = nn.Linear(hidden_size, memory_vector_size)

        # Parameters for gating (write gate controls overall intensity, alloc gate mixes modes)
        self.write_gate_proj = nn.Linear(hidden_size, 1) # g^w in paper
        self.alloc_gate_proj = nn.Linear(hidden_size, 1) # g^a in paper

    def _calculate_allocation_weights(self, usage: torch.Tensor) -> torch.Tensor:
        """Calculates allocation weights based on memory usage.
           Slots with low usage get higher weights. Follows DNC paper Appendix B.
        Args:
            usage: Memory usage vector (batch_size, memory_slots), values in [0, 1].
        Returns:
            allocation_weights: (batch_size, memory_slots)
        """
        batch_size = usage.size(0)
        device = usage.device

        # Ensure usage is non-increasing (this isn't strictly necessary if usage update is correct)
        # usage = torch.cumprod(usage, dim=1) # Optional stability measure

        # Calculate free list scores phi (1 - usage)
        free_list_scores = 1.0 - usage

        # Sort usage to find free locations. Need indices for sorting.
        # Paper uses a stable sort, but torch.sort is generally stable.
        # `sorted_usage` is not directly used, only the indices `free_indices`.
        sorted_usage, free_indices = torch.sort(usage, dim=1) # Ascending sort (low usage first)

        # Calculate allocation weights based on sorted free list (phi)
        # allocation_weights[free_indices[b, j]] = (1 - usage[b, free_indices[b, j]]) * product(usage[b, free_indices[b, k]] for k < j)
        # This is complex to implement directly with scatter/gather.
        # Let's compute the product term efficiently.
        # cumprod_usage needs to be computed on the *sorted* usage.
        cumprod_sorted_usage = torch.cumprod(sorted_usage, dim=1)

        # Pad cumprod_sorted_usage with 1 at the beginning for the j=0 case product
        padded_cumprod = torch.cat([torch.ones(batch_size, 1, device=device), cumprod_sorted_usage[:, :-1]], dim=1)

        # Calculate weights in sorted order
        alloc_weights_sorted = free_list_scores.gather(1, free_indices) * padded_cumprod

        # Unsort the weights back to original memory order using scatter
        # Create template indices for scatter
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(free_indices)
        # Use scatter_: self[batch_indices, free_indices] = alloc_weights_sorted
        allocation_weights = torch.zeros_like(usage)
        allocation_weights.scatter_(dim=1, index=free_indices, src=alloc_weights_sorted)

        return allocation_weights

    def forward(self,
                h: torch.Tensor,
                memory: torch.Tensor,
                prev_usage: torch.Tensor, # Shape: (B, N) - Needed for allocation
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates write parameters and returns instructions for memory update.

        Args:
            h: Controller hidden state (batch_size, hidden_size)
            memory: Memory tensor (batch_size, memory_slots, memory_vector_size)
            prev_usage: Previous usage vector (B, N)

        Returns:
            write_weights: Final weights for writing (B, N)
            erase_vector: Vector for erasing memory (B, M)
            add_vector: Vector for adding to memory (B, M)
            allocation_weights: Weights based on free space (B, N) - useful for usage update
        """
        batch_size = h.size(0)
        device = h.device

        # --- 1. Generate keys, strengths, gates, vectors ---
        write_key = self.key_proj(h) # (B, M)
        write_strength = F.softplus(self.strength_proj(h)) # (B, 1)
        erase_vec = torch.sigmoid(self.erase_vec_proj(h)) # (B, M), values in [0, 1]
        add_vec = self.add_vec_proj(h) # (B, M), linear activation usually
        write_gate = torch.sigmoid(self.write_gate_proj(h)) # (B, 1), g^w
        alloc_gate = torch.sigmoid(self.alloc_gate_proj(h)) # (B, 1), g^a

        # --- 2. Calculate Content-Based Write Weights ---
        # similarity: (B, 1, M) @ (B, M, N) -> (B, N)
        similarity = cosine_similarity(write_key.unsqueeze(1), memory).squeeze(1)
        content_logits = write_strength * similarity # (B, N)
        content_weights = F.softmax(content_logits, dim=-1) # (B, N), w^c_t

        # --- 3. Calculate Allocation Weights ---
        # Based on previous usage vector
        allocation_weights = self._calculate_allocation_weights(prev_usage) # (B, N), w^a_t

        # --- 4. Combine Weights using Gates ---
        # w^w_t = g^w_t [g^a_t w^a_t + (1 - g^a_t) w^c_t]
        write_weights = write_gate * (alloc_gate * allocation_weights + (1 - alloc_gate) * content_weights)

        # --- 5. Return Write Instructions ---
        # The actual memory update happens in the main DNC loop
        return write_weights, erase_vec, add_vec, allocation_weights # Also return alloc_weights for usage calc
