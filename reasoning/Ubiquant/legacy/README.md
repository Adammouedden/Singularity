# Reasoning Module

Four files implementing the Universal Reasoning Model (URM) — a recurrent transformer that iteratively refines a hidden state across multiple loop steps, used for tasks like ARC-AGI and Sudoku.

---

## `common.py`

### `trunc_normal_init_(tensor, std, lower, upper)`
Fills a tensor in-place with truncated normal values using the mathematically correct JAX/Flax formulation. PyTorch's built-in `trunc_normal_` does not produce the correct standard deviation — this function fixes that. Used as the default weight initializer throughout the codebase.

---

## `layers.py`

Core neural network primitives.

### Helpers

| Function | Description |
|---|---|
| `_find_multiple(a, b)` | Rounds `a` up to the nearest multiple of `b`. Used to size MLP intermediate dimensions. |
| `rotate_half(x)` | Splits the last dimension in half and rotates — the core operation of RoPE. |
| `apply_rotary_pos_emb(q, k, cos, sin)` | Applies rotary position embeddings to query and key tensors of shape `[bs, seq_len, heads, head_dim]`. |
| `rms_norm(hidden_states, variance_epsilon)` | Functional RMS normalization with no learned scale. Upcasts to float32 for numerical stability, then returns the original dtype. |

### Classes

#### `CastedLinear`
A `nn.Linear` replacement that casts its weights to match the input dtype at forward time. Enables mixed-precision training without storing multiple weight copies. Initialized with truncated normal; bias is zero-initialized if enabled.

#### `CastedEmbedding`
An embedding table that casts to a fixed target dtype on every lookup. Initialized with truncated normal.

#### `RotaryEmbedding`
Precomputes and caches `cos`/`sin` position tables up to `max_position_embeddings`. `forward()` returns the cached `(cos, sin)` pair.

#### `Attention`
Multi-head self-attention using Flash Attention (`flash_attn_func`). Fuses Q, K, and V into a single projection, applies RoPE, runs Flash Attention, then projects the output. Supports grouped-query attention (GQA) via separate `num_heads` and `num_key_value_heads`. Compatible with both FA2 and FA3 return signatures.

#### `SwiGLU`
Standard SwiGLU FFN block: fused gate+up projection → SiLU activation → optional dropout → down projection. Intermediate size is rounded to a multiple of 256.

#### `ConvSwiGLU`
SwiGLU variant with a depthwise Conv1d inserted after the gate activation. The convolution adds local sequence modeling on top of the standard FFN. This is the MLP used inside every `URMBlock`.

#### `FullyLinearGLU` / `LinearGLU` / `SiLU` / `ReLU` / `LinearSwish`
Alternative FFN variants available in the library. Not used by URM directly.

---

## `sparse_embedding.py`

Sparse embedding infrastructure for per-puzzle learned embeddings, designed for efficient training when only a small batch of embeddings are updated per step.

### `CastedSparseEmbedding`

Stores the full embedding table in `weights` (a non-gradient buffer). At training time, copies the batch's looked-up rows into `local_weights` — a small gradient buffer of size `[batch_size, emb_dim]` — so gradients only flow through the active slice, not the full table. At inference time, indexes `weights` directly with no gradient overhead. Validates index bounds on every forward call.

### `CastedSparseEmbeddingSignSGD_Distributed`

Custom optimizer for `CastedSparseEmbedding`. Uses SignSGD (gradient sign only) with decoupled weight decay. Each param group must contain exactly three params: `local_weights` (requires grad), `local_ids` (1D int), and `weights` (2D full table). When `world_size > 1`, all-gathers gradients and IDs across ranks before applying the update, then writes results back into the sparse rows of `weights` via `scatter_add_` and unique index deduplication.

### `_sparse_emb_signsgd_dist(...)` *(internal)*
Implements the full sparse update: all-gather → deduplicate indices → compute SignSGD step → scatter rows back into the weight table.

---

## `urm/URM.py`

The Universal Reasoning Model: a recurrent transformer that iteratively refines a hidden state, deciding per-example when to halt.

### `URMConfig`

Pydantic model holding all hyperparameters.

| Field | Description |
|---|---|
| `batch_size`, `seq_len` | Input dimensions |
| `puzzle_emb_ndim` | Dimensionality of per-puzzle embeddings. Set to `0` to disable. |
| `num_puzzle_identifiers` | Number of distinct puzzle IDs |
| `vocab_size` | Token vocabulary size |
| `hidden_size` | Transformer hidden dimension |
| `num_heads` | Number of attention heads. `hidden_size` must be divisible by `num_heads`. |
| `expansion` | MLP intermediate size multiplier |
| `loops` | Maximum number of outer recurrent steps |
| `L_cycles` | Number of full transformer passes per loop step |
| `H_cycles` | Additional no-grad burn-in passes per loop step |
| `forward_dtype` | Compute dtype string (default: `"bfloat16"`) |

### `URMCarry`

Dataclass holding the full recurrent state passed between loop iterations.

| Field | Shape | Description |
|---|---|---|
| `current_hidden` | `[B, seq_len, hidden_size]` | The running hidden state |
| `steps` | `[B]` int32 | Number of loop steps taken per example |
| `halted` | `[B]` bool | Whether each example just completed a loop step |
| `current_data` | `Dict[str, Tensor]` | Frozen copy of the active batch per example |

### `URMBlock`

One transformer block: Flash Attention residual → RMSNorm → ConvSwiGLU residual → RMSNorm. Norm is applied post-residual (not pre-norm).

### `URM_Inner`

The core transformer. Owns all learned parameters.

| Method | Description |
|---|---|
| `_input_embeddings(input, puzzle_identifiers)` | Looks up token embeddings, optionally prepends puzzle embeddings (padded to fit `hidden_size`), and scales by `√hidden_size`. |
| `empty_carry(batch_size)` | Allocates an uninitialized `current_hidden` of the correct shape and dtype. |
| `reset_carry(reset_flag, carry)` | Replaces `current_hidden` with the learned `init_hidden` vector for any example where `reset_flag=True`. |
| `forward(carry, batch)` | Runs `H_cycles - 1` no-grad burn-in passes then one grad-tracked pass of `L_cycles` transformer layers. Returns `(new_carry, token_logits, (q_halt_logits, q_continue_logits))`. |

`lm_head` maps hidden states to token logits. `q_head` produces two scalar logits per example used for the halt/continue decision.

### `URM`

The outer recurrent loop wrapping `URM_Inner`.

| Method | Description |
|---|---|
| `initial_carry(batch)` | Creates the starting `URMCarry`: zeroed steps, all `halted=True` (forces a reset on the first step), and empty `current_data`. |
| `forward(carry, batch, compute_target_q)` | Executes one recurrent step. Resets carry and loads fresh batch data for halted examples. Calls `URM_Inner`, then computes updated `halted` flags. During training with `loops > 1`, uses `q_halt_logits` and random exploration for early stopping. Returns `(new_carry, outputs_dict)` where `outputs_dict` contains `logits`, `q_halt_logits`, and `q_continue_logits`. |
