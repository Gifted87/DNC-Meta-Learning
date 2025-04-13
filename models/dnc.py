
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, List, Optional

from .memory_heads import MemoryReadHead, MemoryWriteHead
from .utils import outer_product

class DNC(nn.Module):
    """
    Differentiable Neural Computer (DNC) implementation.
    Combines a controller (LSTM) with an external memory accessed via attention heads.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_actions: int, # Output dimension for policy head
                 memory_slots: int = 64,
                 memory_vector_size: int = 32,
                 num_read_heads: int = 3,
                 k_sparse_read: Optional[int] = None # Top-k for sparse read (default ~5%)
                ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.memory_slots = memory_slots
        self.memory_vector_size = memory_vector_size
        self.num_read_heads = num_read_heads

        if k_sparse_read is None:
            # Default sparse read k to ~5% of memory slots, minimum 1
            self.k_sparse_read = max(1, memory_slots // 20)
        else:
            self.k_sparse_read = max(1, k_sparse_read)

        # Controller (LSTM Cell)
        controller_input_dim = input_size + num_read_heads * memory_vector_size
        self.controller = nn.LSTMCell(controller_input_dim, hidden_size)

        # Memory Heads
        self.read_heads = nn.ModuleList([
            MemoryReadHead(
                memory_slots, memory_vector_size, hidden_size, num_read_heads, self.k_sparse_read
            ) for _ in range(num_read_heads)
        ])
        self.write_head = MemoryWriteHead(memory_slots, memory_vector_size, hidden_size)

        # Output heads
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1) # For Actor-Critic / PPO

        # Optional: Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Simple initialization (can be customized)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0) # Or Kaiming for ReLU-like
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                for name, param in module.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                        # Initialize forget gate bias to 1 for better gradient flow initially
                        # Bias tensor shape: (4 * hidden_size), gates order: i, f, g, o
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        nn.init.constant_(param[start:end], 1.)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)


    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initializes the DNC state dictionary for a new sequence."""
        # Controller state (hidden and cell)
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        controller_state = (h, c)

        # Memory state (Initialize to near-zero small random values or zeros)
        memory = torch.randn(batch_size, self.memory_slots, self.memory_vector_size, device=device) * 0.01
        # memory = torch.zeros(batch_size, self.memory_slots, self.memory_vector_size, device=device)

        # Interface vectors (weights, usage, links) - Initialize to zeros or uniform/small values
        # Initialize read weights almost uniform but slightly focused on first slot?
        read_weights = torch.zeros(batch_size, self.num_read_heads, self.memory_slots, device=device)
        read_weights[:, :, 0] = 1.0 # Slight focus on first slot initially
        # read_weights = F.softmax(torch.ones(batch_size, self.num_read_heads, self.memory_slots, device=device) * 0.1, dim=-1) # Alternative: near uniform

        write_weights = torch.zeros(batch_size, self.memory_slots, device=device)
        usage = torch.zeros(batch_size, self.memory_slots, device=device) # Start with zero usage
        link_matrix = torch.zeros(batch_size, self.memory_slots, self.memory_slots, device=device) # No links initially
        precedence_weights = torch.zeros(batch_size, self.memory_slots, device=device) # Needed for link update

        return {
            "controller": controller_state,
            "memory": memory,
            "read_weights": read_weights,     # (B, num_read_heads, N)
            "write_weights": write_weights,   # (B, N)
            "usage": usage,                   # (B, N)
            "link_matrix": link_matrix,       # (B, N, N)
            "precedence_weights": precedence_weights # (B, N)
        }

    def _update_usage(self,
                      prev_usage: torch.Tensor,
                      write_weights: torch.Tensor,
                      read_weights: torch.Tensor # Shape (B, num_read_heads, N)
                     ) -> torch.Tensor:
        """Updates the usage vector based on DNC paper Eq. 8."""
        # Ensure read_weights are detached if they come from a previous step's graph?
        # Usage calculation depends on current write and *previous* reads (or reads just calculated).
        # Let's assume read_weights are the ones just calculated by the read heads.
        with torch.no_grad(): # Usage calculation shouldn't propagate gradients back to read heads directly
            # Calculate retention psi: product over read heads of (1 - read_weight_i)
            psi = torch.prod(1.0 - read_weights, dim=1) # (B, N)

        # Update usage: u_t = (u_{t-1} + w^w_{t-1} - u_{t-1} * w^w_{t-1}) * psi_t
        # Note: Paper uses w^w_{t-1}, but often implemented with current w^w_t for simplicity. Let's use current.
        # usage_increase = prev_usage + write_weights - prev_usage * write_weights # Effect of writing
        # updated_usage = usage_increase * psi # Effect of reading (retention)

        # Alternative from some implementations: Usage is simply discounted prev_usage * psi
        # Seems less correct than the paper's formula. Let's try the paper's logic.
        # We need prev_write_weights here based on Eq 8!
        # Let's assume the state dict holds prev_write_weights.
        # **Decision:** Modify state/forward to pass prev_write_weights for correct usage.
        # **Revised Logic (assuming prev_write_weights available):**
        # usage_retention = prev_usage * psi # Decay based on reads
        # usage_after_write = usage_retention + prev_write_weights - usage_retention * prev_write_weights # Add effect of *previous* write
        # return usage_after_write

        # **Simpler Implementation (Common Variation):** Use current write weights. Less faithful but works.
        updated_usage = (prev_usage + write_weights - prev_usage * write_weights) * psi.detach() # Detach psi?
        return updated_usage


    def _update_link_matrix(self,
                            prev_link_matrix: torch.Tensor, # (B, N, N)
                            prev_precedence_weights: torch.Tensor, # (B, N)
                            write_weights: torch.Tensor # Current write weights (B, N)
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the temporal link matrix and precedence weights (Eq. 6, 7)."""
        batch_size, num_slots = write_weights.shape
        device = write_weights.device

        # Calculate current precedence weights p_t (Eq. 6)
        # p_t = (1 - sum(w^w_t)) * p_{t-1} + w^w_t
        write_sum = torch.sum(write_weights, dim=1, keepdim=True) # (B, 1)
        current_precedence_weights = (1.0 - write_sum) * prev_precedence_weights + write_weights # (B, N)

        # Update link matrix L_t (Eq. 7)
        # L_t = (1 - w^w_t * 1^T - 1 * (w^w_t)^T) * L_{t-1} + w^w_t * (p_{t-1})^T
        ww_col = write_weights.unsqueeze(2) # (B, N, 1)
        ww_row = write_weights.unsqueeze(1) # (B, 1, N)
        prev_p_row = prev_precedence_weights.unsqueeze(1) # (B, 1, N)

        # Decay factor matrix (1 - w^w_t * 1^T - 1 * (w^w_t)^T)
        # Careful with broadcasting: need elementwiseodot
        decay_factor = 1.0 - ww_col - ww_row # (B, N, N) - wrong shapes, need outer product concept
        # Let's use outer_product utils if possible? No, this is elementwise for L_{t-1}
        # Try again: Expand w^w_t to (B, N, N)
        ww_col_expanded = ww_col.expand(-1, -1, num_slots)
        ww_row_expanded = ww_row.expand(-1, num_slots, -1)
        decay_matrix = (1.0 - ww_col_expanded) * (1.0 - ww_row_expanded) # Close to elementwise (1 - w_i)(1 - w_j)

        # Apply decay and add new links
        updated_link_matrix = decay_matrix * prev_link_matrix
        # Add new links: w^w_t * (p_{t-1})^T -> outer product
        new_links = outer_product(write_weights, prev_precedence_weights) # (B, N, N)
        updated_link_matrix += new_links

        # Ensure diagonal is zero (link from i to i is not meaningful)
        # Create diagonal mask
        diag_mask = torch.eye(num_slots, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        updated_link_matrix = updated_link_matrix * (1.0 - diag_mask)

        return updated_link_matrix, current_precedence_weights


    def forward(self,
                x: torch.Tensor,
                prev_state: Dict[str, torch.Tensor]
               ) -> Tuple[Categorical, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs one step of DNC computation.

        Args:
            x: Input tensor (batch_size, input_size)
            prev_state: Dictionary containing the previous DNC state.

        Returns:
            action_dist: Categorical distribution over actions.
            value: Estimated state value.
            new_state: Dictionary containing the updated DNC state.
        """
        device = x.device
        x = x.to(device)
        batch_size = x.size(0)

        # --- Unpack previous state and ensure tensors are on the correct device ---
        prev_controller_state = tuple(s.to(device) for s in prev_state["controller"])
        prev_memory = prev_state["memory"].to(device)
        prev_read_weights = prev_state["read_weights"].to(device) # (B, num_read_heads, N)
        prev_write_weights = prev_state["write_weights"].to(device) # (B, N) - Needed for usage/link update
        prev_usage = prev_state["usage"].to(device) # (B, N)
        prev_link_matrix = prev_state["link_matrix"].to(device) # (B, N, N)
        prev_precedence_weights = prev_state["precedence_weights"].to(device) # (B, N)

        # --- 1. Read from memory ---
        read_vectors: List[torch.Tensor] = []
        current_read_weights_list: List[torch.Tensor] = []
        prev_h, _ = prev_controller_state # Read heads often depend on previous hidden state
        for i in range(self.num_read_heads):
            # Pass necessary state components to read head
            # Requires: h, memory. Optional: prev_read_weights (for *this* head), link_matrix
            # Passing all prev_read_weights, head can index if needed (though current head doesn't)
            r_i, w_r_i = self.read_heads[i](prev_h, prev_memory, prev_read_weights, prev_link_matrix)
            read_vectors.append(r_i)
            current_read_weights_list.append(w_r_i)

        current_read_weights = torch.stack(current_read_weights_list, dim=1) # (B, num_read_heads, N)

        # --- 2. Controller step ---
        controller_input = torch.cat([x] + read_vectors, dim=-1)
        h, c = self.controller(controller_input, prev_controller_state)

        # --- 3. Write to memory ---
        # Write head needs: h, memory, prev_usage
        write_w, erase_v, add_v, alloc_w = self.write_head(h, prev_memory, prev_usage)
        # Perform the memory update using returned instructions
        erase_matrix = outer_product(write_w, erase_v) # (B, N, M)
        add_matrix = outer_product(write_w, add_v) # (B, N, M)
        updated_memory = prev_memory * (1.0 - erase_matrix) + add_matrix

        # --- 4. Update Usage and Link Matrix ---
        # Usage depends on previous usage, current write weights, and current read weights
        # **Correction:** Usage depends on PREVIOUS write weights (w^w_{t-1}) per paper.
        current_usage = self._update_usage(prev_usage, prev_write_weights, current_read_weights)

        # Link matrix depends on previous link matrix, previous precedence weights, and current write weights
        updated_link_matrix, current_precedence_weights = self._update_link_matrix(
            prev_link_matrix, prev_precedence_weights, write_w
        )

        # --- 5. Output - Policy and Value ---
        logits = self.policy_head(h)
        value = self.value_head(h)
        action_dist = Categorical(logits=logits)

        # --- 6. Assemble new state dictionary ---
        new_state = {
            "controller": (h, c),
            "memory": updated_memory,
            "read_weights": current_read_weights,
            "write_weights": write_w, # Current write weights become next step's prev_write_weights
            "usage": current_usage,
            "link_matrix": updated_link_matrix,
            "precedence_weights": current_precedence_weights
        }

        return action_dist, value, new_state
