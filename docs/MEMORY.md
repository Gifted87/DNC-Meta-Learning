# Differentiable Memory Math (DNC Implementation Notes)

This document outlines core mathematical concepts behind the DNC's memory system, reflecting the choices made in this implementation. See the original [DeepMind DNC paper](https://www.nature.com/articles/nature20101) for full details.

## 1. Content-Based Addressing (Read & Write)

- **Keys & Strength:** The controller (`h_t`) outputs a key (`k_t`, size M) and strength (`β_t > 0`) via linear projections.
- **Similarity:** Batched [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) `C(k_t, M_{t-1}[i])` is calculated between the key and each memory slot `M_{t-1}[i]`.
  $$ C(u, v) = \frac{u \cdot v}{\|u\| \|v\|} $$
- **Content Weights (`w^c_t`)**: A softmax is applied over the scaled similarities:
  $$ w^c_t[i] = \frac{\exp(\beta_t C(k_t, M_{t-1}[i]))}{\sum_j \exp(\beta_t C(k_t, M_{t-1}[j]))} $$
- **Sparse Reads:** For read heads, a sparse softmax (`sparse_softmax_topk`) is used instead, focusing probability mass on the `k` memory locations with the highest `β_t C(k_t, M_{t-1}[i])` scores, setting others to zero. Content weights for writing (`w^c_t` used in write head) use standard softmax.

## 2. Allocation Weighting (Write Head)

To write to unused memory, allocation weights (`w^a_t`) are calculated based on the *previous* step's usage vector (`u_{t-1}`). Low usage implies freeness.

- **Usage (`u_t`):** Represents the degree to which each slot is "occupied" (value in [0, 1]). Updated after reads/writes (see below).
- **Free List (`phi`):** Calculated as `phi = 1 - u_{t-1}`.
- **Sorting:** Usage `u_{t-1}` is sorted ascendingly to get indices `free_indices` mapping sorted order back to original slots.
- **Allocation Calculation (Appendix B, DNC Paper):** Weights are assigned based on freeness and the product of usage of previously considered free slots.
  $$ w^a_t[\text{free}_j] = (1 - u_{t-1}[\text{free}_j]) \prod_{k=1}^{j-1} u_{t-1}[\text{free}_k] $$
  Where `free_j` is the index of the j-th freest slot. This implementation calculates this efficiently using `torch.sort` and `torch.cumprod`, then unsorts using `scatter_`.

## 3. Write Weight Combination

The final write weights (`w^w_t`, size N) combine allocation and content weights using gates (`g^w_t`, `g^a_t` in [0, 1]) from the controller:

$$ w^w_t = g^w_t \left[ g^a_t w^a_t + (1 - g^a_t) w^c_t \right] $$

## 4. Memory Update

Writing uses the write weights (`w^w_t`), an erase vector (`e_t`, size M, values in [0, 1]), and an add vector (`a_t`, size M) from the controller.

$$ M_t = M_{t-1} \odot (\mathbf{1} - w^w_t \otimes e_t^T) + w^w_t \otimes a_t^T $$

Where `⊙` is element-wise multiplication, `⊗` is outer product, and `1` is a matrix of ones. The `(1 - ...)` term performs the erase operation based on write weights and the erase vector.

## 5. Usage Vector Update (`u_t`)

Usage reflects memory occupation and is updated based on current reads and *previous* writes.

- **Read Retention (`psi_t`):** Measures how much usage is retained after reading. Calculated based on *current* read weights (`w^r_t`, shape B x num_heads x N).
  $$ \psi_t = \prod_{h=1}^{\text{num_reads}} (1 - w^r_{t, h}) $$
- **Usage Update (Eq. 8, DNC Paper):** Combines previous usage decayed by reading (`u_{t-1} * \psi_t`) and the effect of the *previous* write (`w^w_{t-1}`).
  $$ u_t = (u_{t-1} + w^w_{t-1} - u_{t-1} \odot w^w_{t-1}) \odot \psi_t $$
  *(Note: This implementation uses the previous step's write weights `prev_write_weights` stored in the DNC state).*

## 6. Temporal Link Matrix (`L_t`)

Tracks the order of writes (N x N matrix). `L_t[i, j] \approx 1` implies `i` was written immediately after `j`.

- **Precedence Weights (`p_t`):** Probability that slot `i` was the *last* one written to. Updated based on *current* write weights (`w^w_t`) and previous precedence weights (`p_{t-1}`).
  $$ p_t = (1 - \sum_j w^w_t[j]) p_{t-1} + w^w_t $$
- **Link Matrix Update (Eq. 7, DNC Paper):** Decays previous links and adds new links based on current writes (`w^w_t`) and previous precedence (`p_{t-1}`).
  $$ L_t = (1 - w^w_t \mathbf{1}^T - \mathbf{1} (w^w_t)^T) \odot L_{t-1} + w^w_t \otimes (p_{t-1})^T $$
  *(Note: The decay term is implemented elementwise as `(1 - w^w_{t,i})(1 - w^w_{t,j}) * L_{t-1}[i, j]`. The diagonal is zeroed out.)*

## 7. Read Modes (Simplified)

The full DNC allows read heads to interpolate between content lookup, forward temporal links, and backward temporal links. This implementation currently simplifies reading:

- **Content:** Uses sparse content lookup (`w^c_t`).
- **Temporal:** Currently *not* implemented in the read heads (marked TODO). A full implementation would use `L_t` and `p_t` with gating mechanisms controlled by the controller.
- **Final Read Weights (`w^r_t`):** Currently set directly to the sparse content weights.
