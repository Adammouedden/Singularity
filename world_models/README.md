# Vision Language World Model
We propose using a VLWM to create a prototypable reasoning model powered by world understanding.


# Research Engineers
Fernando Crespo Vazquez
David Orjuela

---

## Overview

The `world_models` module implements three interchangeable **action proposers**. Each proposer receives the current environment state and the list of available actions, then returns the 3 best candidate next actions for the MCTS planner to evaluate.

```
MCTS Planner
    │
    ├── proposer.propose_actions(state, actions)
    │       │
    │       ├── CNNWorldModel        (ARC visual tasks — learns online from experience)
    │       ├── GeminiWorldModel     (ARC visual tasks — Gemini LLM reasoning)
    │       └── GeneralWorldModel    (Math / logic / text tasks — Gemini LLM reasoning)
    │
    └── Returns: CandidateActions (3 ranked ActionCandidate objects)
```

All three models implement the same interface and can be swapped into the MCTS search with no other changes.

---

## Role in the Singularity Pipeline

```
Environment State (EnvState)
  frame: 64×64 int grid   score: float   available_actions: [1..7]
        │
        ▼
┌──────────────────────────┐
│      World Model         │  ← proposes 3 candidate actions
│  (CNN / Gemini / General)│
└────────────┬─────────────┘
             │  CandidateActions
             ▼
┌──────────────────────────┐
│         Critic           │  ← scores each candidate
│  (Gemini / General)      │
└────────────┬─────────────┘
             │  ranked (action, score) pairs
             ▼
┌──────────────────────────┐
│       MCTS Planner       │  ← selects and commits best action
│  (search/mcts.py)        │
└──────────────────────────┘
```

---

## World Models

### `CNNWorldModel` — `CNN_world_model.py`

A locally-trained convolutional network that **learns which actions cause state changes** as the agent plays. No API calls — runs entirely on-device.

#### Network Architecture

```
Input: (B, 16, 64, 64)   ← one-hot encoded color channels for 64×64 ARC grid
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Shared Convolutional Backbone                             │
│  Conv2d(16→32) → Conv2d(32→64) → Conv2d(64→128) →         │
│  Conv2d(128→256)   [all with BatchNorm + ReLU]             │
└──────────────────────────┬─────────────────────────────────┘
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
  ┌─────────────────────┐    ┌────────────────────────────┐
  │    Action Head      │    │     Coordinate Head         │
  │  MaxPool(4×4)       │    │  Conv2d(256→128→64→32→1)   │
  │  FC(65536→512)      │    │  Flatten → 4096 logits     │
  │  Dropout(0.2)       │    │  (one per grid cell)       │
  │  FC(512→5)          │    └────────────┬───────────────┘
  │  5 action logits    │                 │
  └──────────┬──────────┘                 │
             └──────────┬─────────────────┘
                        ▼
            Combined logits: (B, 4101)
            [5 action types + 4096 coordinates]
```

#### Action Space

| Index | Action | Meaning |
|-------|--------|---------|
| 0–4   | ACTION1–5 | up / down / left / right / special |
| 5–4100 | ACTION6 (x,y) | coordinate click on 64×64 grid |

#### Online Learning Loop

```
Agent takes action
        │
        ▼
observe_action(prev_state, action, next_state)
  1. Detect if frame changed  →  reward = 1.0 / 0.0
  2. Hash (frame, action) for deduplication
  3. Add to experience buffer (max 200,000 transitions)
  4. Every 5 actions → sample batch of 64 → train
        │
        ▼
_train_action_model()
  Loss = BCE(predictions, frame_changed) 
       − 0.0001 × action_entropy 
       − 0.00001 × coord_entropy
  Optimizer: Adam (lr=0.0001)
```

**Per-level reset:** When `score` increases (new puzzle), the network and buffer reset completely so the model adapts fresh to each level.

---

### `GeminiWorldModel` — `gemini_world_model.py`

Uses Gemini 2.5 Flash to reason about ARC visual frames. Converts raw grids into semantic abstractions before prompting, and caches results to avoid redundant API calls.

#### Frame Abstraction Pipeline

```
Raw frame (64×64 int grid)
        │
        ▼
frame_to_key(frame)           → immutable hashable key
        │
        ▼
extract_frame_abstraction()   → structured representation (no raw cells)
        │
        ▼
summarize_for_llm()           → semantic text description
        │
        ▼
Gemini 2.5 Flash prompt       → structured JSON (3 ActionCandidate objects)
```

#### Caching

| Cache | Key | Value |
|-------|-----|-------|
| `frame_summary_cache` | `frame_key` | Semantic LLM summary string |
| `proposal_cache` | `(frame_key, actions_tuple)` | Full `CandidateActions` result |

Cache hits skip all LLM calls and return a deep copy of the stored result.

---

### `GeneralWorldModel` — `general_world_model.py`

Uses Gemini 2.5 Flash (temperature=0.2) for non-visual reasoning tasks such as math and logic problems.

#### Universal Action Set

| Action | Meaning |
|--------|---------|
| `UNDERSTAND` | Parse and internalize the problem |
| `TRANSFORM` | Rewrite or restructure the expression |
| `SOLVE_SUBPART` | Isolate and solve one component |
| `VERIFY` | Check a result against constraints |
| `FINALIZE` | Produce the final answer |

These abstract actions are domain-agnostic and map to high-level reasoning steps rather than low-level coordinates.

**Example frame format:**
```python
{
    "problem": "Solve for x: 2x + 5y = 17 − 9.  Given y = 5",
    "working": "",
    "final_answer": None
}
```

---

## Shared Data Schemas

```python
class EnvState(BaseModel):
    frame: List[List[int]]        # 64×64 color grid (values 0–15 for ARC)
    state: str                    # "NOT_FINISHED" | "WIN" | "GAME_OVER"
    score: float                  # Current puzzle score / level progress
    available_actions: List[int]  # e.g. [1, 2, 3, 4, 5, 6]
    step_index: int
    guid: Optional[str]
    game_id: Optional[str]
    card_id: Optional[str]

class ActionCandidate(BaseModel):
    action: str                   # "ACTION1" … "ACTION7"
    x: Optional[int]              # Grid column [0–63], only for ACTION6
    y: Optional[int]              # Grid row    [0–63], only for ACTION6
    rationale: Optional[str]

class CandidateActions(BaseModel):
    candidates: List[ActionCandidate]   # Always 3 items
```

---

## File Reference

| File | Purpose |
|------|---------|
| `CNN_world_model.py` | CNN action proposer with online learning (ARC) |
| `gemini_world_model.py` | Gemini LLM proposer with frame abstraction + caching (ARC) |
| `general_world_model.py` | Gemini LLM proposer for math/logic tasks |

---

## Comparison

| | CNNWorldModel | GeminiWorldModel | GeneralWorldModel |
|--|---------------|-----------------|-------------------|
| **Domain** | ARC visual | ARC visual | Math / logic / text |
| **Learns?** | Yes — online per level | No | No |
| **Backend** | Local PyTorch CNN | Gemini 2.5 Flash API | Gemini 2.5 Flash API |
| **Speed** | ~ms (GPU/CPU) | ~1–2 s per call | ~1–2 s per call |
| **Input** | Raw 64×64 grid tensor | Grid → semantic summary | Problem state dict |
| **Action space** | Discrete (5) + coordinate grid (4096) | Discrete (7) | 5 universal actions |
| **Caching** | Experience buffer | Frame + proposal caches | None |
