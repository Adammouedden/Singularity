# Singularis

**Singularis** is a reasoning-augmented sequence-to-sequence model built on T5Gemma2. It inserts a **Unified Reasoning Module (URM)** between the encoder and decoder, giving the model an explicit iterative reasoning stage before generating output.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                    INPUT                         │
│         Text tokens  +  (optional) Images        │
└────────────────────────┬─────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │         T5Gemma2 Encoder        │
         │                                │
         │  • 18-layer transformer        │
         │  • Alternating sliding-window  │
         │    (512-token) + full attention │
         │  • SigLip vision tower (27 L)  │
         │    for 896×896 image inputs    │
         │  • Multi-modal projector       │
         │                                │
         │  Output: [B, S, 640]           │
         │         (frozen during URM     │
         │          training)             │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │          URMBridge             │
         │     (Unified Reasoning Module) │
         │                                │
         │  • 4 transformer blocks        │
         │  • 16 reasoning loops          │
         │  • Encoder output injected     │
         │    as signal every cycle       │
         │  • No projection layers —      │
         │    hidden_size 640 is shared   │
         │    with encoder & decoder      │
         │                                │
         │  Output: [B, S, 640]           │
         │         (trained; only         │
         │          trainable component)  │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │        T5Gemma2 Decoder        │
         │                                │
         │  • 18-layer transformer        │
         │  • Cross-attention over URM    │
         │    refined hidden states       │
         │  • KV-cache for fast inference │
         │  • Autoregressive generation   │
         │                                │
         │  Output: [B, S_dec, 262144]    │
         │         (frozen during basic   │
         │          training; LoRA in     │
         │          enhanced training)    │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │        lm_head (Linear)        │
         │        640 → 262,144           │
         │   Next-token prediction logits │
         └────────────────────────────────┘
```

---

## The URM — What It Does

The URM is the core innovation. Instead of passing encoder hidden states directly to the decoder, Singularis runs them through **16 loops of iterative refinement**:

```
Encoder Output [B, S, 640]
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Loop 1                                           │
│  ┌─────────────────────────────────────────────┐  │
│  │  Transformer Block × 4                     │  │
│  │  (Self-Attention + SwiGLU MLP)              │  │
│  │  Encoder output injected as input signal   │  │
│  └──────────────────────────┬──────────────────┘  │
│                             │                      │
│  Loop 2 → 16  (same blocks, updated carry state)  │
└──────────────────────────────┬────────────────────┘
                               │
                Refined [B, S, 640]  →  Decoder
```

**Key design decisions:**
- `hidden_size=640` is identical across encoder, URM, and decoder — no projection needed
- URM has no token embedding table; encoder hidden states are fed in directly
- `loops=16` cycles give the URM the equivalent of a "thinking" phase
- `L_cycles=8` and `H_cycles=2` govern internal adaptive halting

---

## Model Dimensions

| Component | Layers | Hidden Size | Heads | KV Heads |
|-----------|--------|-------------|-------|----------|
| Encoder   | 18     | 640         | 4     | 1        |
| URM       | 4      | 640         | 8     | —        |
| Decoder   | 18     | 640         | 4     | 1        |
| Vision (SigLip) | 27 | 1152     | 16    | —        |

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~270M |
| Vocabulary Size | 262,144 |
| Max Sequence Length | 32,768 |
| URM Reasoning Loops | 16 |
| URM Transformer Blocks | 4 |
| Head Dimension | 256 |
| Intermediate Size | 2,048 |
| Image Resolution | 896×896 |
| Patch Size | 14×14 |

---

## Repository Structure

```
singularis/
├── model.py                        # Singularis & SingularisForConditionalGeneration
├── urm_bridge.py                   # URMBridge — wraps URM, handles carry state
├── config_and_weights.py           # Loads configs, extracts weights from safetensors
│
├── configs/
│   └── t5gemma2_270M_config.json   # Full T5Gemma2 architecture config
│
├── weights/
│   └── model.safetensors           # Pre-trained encoder + decoder weights
│
├── training/
│   ├── datasets.py                 # GSM8K, ARC-Easy, ARC-Challenge dataloaders
│   ├── train_Singularis.py         # Basic training loop (URM-only)
│   └── train_Singularis_colab.py   # Enhanced training (LoRA + AMP + accumulation)
│
├── untrained_testing/
│   ├── dummy_test_Singularis.py    # Shape/pipeline smoke test
│   └── autoregressive_Singularis.py # Generation inference test
│
└── benchmark/
    ├── singularis_ARC-e.py         # Singularis on ARC-Easy
    ├── singularis_ARC-c.py         # Singularis on ARC-Challenge
    ├── t5gemma2_ARC-e.py           # Baseline (no URM) on ARC-Easy
    ├── t5gemma2_ARC-c.py           # Baseline (no URM) on ARC-Challenge
    └── results/
        ├── ARC-e/                  # ARC-Easy JSON result files
        └── ARC-c/                  # ARC-Challenge JSON result files
```

---

## Core Classes

### `Singularis` — `model.py`

Extends `T5Gemma2Model`. Replaces the standard encoder→decoder path with encoder→URM→decoder.

```python
model = Singularis(urm_config=URM_config, LLM_config=LLM_config)

output = model.forward(
    input_ids=...,            # [B, S]     — text tokens
    pixel_values=...,         # [B, C, H, W] — optional images
    attention_mask=...,       # [B, S]
    decoder_input_ids=...,    # [B, S_dec] — shifted target tokens
)
# Returns: Seq2SeqModelOutput
```

### `SingularisForConditionalGeneration` — `model.py`

Drop-in replacement for `T5Gemma2ForConditionalGeneration`. Use this for training and inference.

```python
model = SingularisForConditionalGeneration(URM_config, LLM_config)
model.load_weights(encoder_weights, decoder_weights)

# Greedy generation
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    do_sample=False,
)
```

### `URMBridge` — `urm_bridge.py`

Wraps the URM and handles the carry-state initialization needed to inject encoder output at each reasoning cycle.

```python
# Input:  encoder hidden states [B, S, 640]
# Output: refined hidden states [B, S, 640]
refined = urm_bridge(encoder_hidden_states)
```

The bridge initializes a fresh carry state at each forward pass (`halted=True` triggers carry reset in the URM), then passes the encoder output as the URM's input signal for all 16 loops.

---

## Training

### Strategy

Training is designed to inject reasoning capability into the URM while keeping the pre-trained encoder and decoder intact:

```
Encoder  ──  FROZEN  (pre-trained T5Gemma2 weights)
URM      ──  TRAINED (randomly initialized → learns to reason)
Decoder  ──  FROZEN  (pre-trained T5Gemma2 weights)
```

### Datasets

| Dataset | Task | Input Format | Target |
|---------|------|--------------|--------|
| **GSM8K** | Grade school math | `solve: {question}` | Full solution chain |
| **ARC-Easy** | Multiple-choice science | `Question: ...\nChoices:\n(A)...\nAnswer:` | Single letter (A/B/C/D) |
| **ARC-Challenge** | Harder multiple-choice science | Same as ARC-Easy | Single letter (A/B/C/D) |

Training uses **weighted random sampling** to balance all three datasets at ~33% each, compensating for their different sizes.

### Basic Training — `training/train_Singularis.py`

```
Optimizer:    AdamW  (lr=1e-3)
Epochs:       50
Early stop:   patience=7
Loss:         CrossEntropyLoss (ignore_index=-100)
Checkpoint:   checkpoints/urm_best.pt  (URM weights only)
Logging:      WandB
```

### Enhanced Training — `training/train_Singularis_colab.py`

Designed for Colab with memory constraints and faster convergence:

```
Optimizer:     AdamW  (lr=2e-4, weight_decay=0.01)
Scheduler:     Linear warmup (100 steps) → linear decay
Precision:     bfloat16 AMP
Grad accum:    4 steps → effective batch size 32
Grad clipping: norm=1.0
LoRA:          r=32, α=64 on decoder q/k/v/o projections
Trainable:     URMBridge + lm_head + LoRA adapter weights
```

The LoRA variant allows the decoder to slightly adapt to URM-processed representations while keeping parameter count low.

---

## Benchmarks

Evaluations compare Singularis (with untrained URM) against the vanilla T5Gemma2 baseline to measure the gap that URM training needs to close.

### ARC-Easy (2,376 test examples)

| Model | Accuracy | Duration |
|-------|----------|----------|
| T5Gemma2-270M (no URM, baseline) | 24.33% | 211.4s |
| Singularis (untrained URM)       | 12.08% | 205.4s |

The ~12% gap is expected — URM weights are random prior to training. The baseline score provides the target that trained Singularis should surpass.

### Evaluation Setup

- Decoding: greedy (`do_sample=False`, `max_new_tokens=10`)
- Dtype: `bfloat16`
- Batch size: 8
- Results saved as JSON with full sample-level outputs

---

## URM Configuration

```python
URMConfig(
    hidden_size   = 640,     # Must match encoder/decoder hidden_size
    num_layers    = 4,       # Transformer blocks inside URM
    num_heads     = 8,       # Self-attention heads
    loops         = 16,      # Iterative reasoning cycles
    L_cycles      = 8,       # Latent reasoning cycles (adaptive halting)
    H_cycles      = 2,       # Hidden state cycles (adaptive halting)
    expansion     = 4.0,     # SwiGLU MLP expansion factor
    seq_len       = 512,     # Max sequence length
    batch_size    = 32,
    forward_dtype = "bfloat16",
    vocab_size    = 1,       # Unused — no token embedding table
)
```
As stated in the URM paper.
