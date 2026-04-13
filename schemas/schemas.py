from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class ActionCandidate(BaseModel):
    action: str = Field(
        description="One of ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
    )
    x: Optional[int] = Field(
        default=None,
        description="x coordinate if action is ACTION6, else null"
    )
    y: Optional[int] = Field(
        default=None,
        description="y coordinate if action is ACTION6, else null"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Short explanation for why this action may help"
    )

class CandidateActions(BaseModel):
    candidates: List[ActionCandidate] = Field(
        description="Top 3 candidate next actions"
    )

class ScoredAction(BaseModel):
    index: int = Field(description="Index of the action in input list")
    score: float = Field(description="Score from 0.0 to 1.0")

class EvaluationResult(BaseModel):
    results: List[ScoredAction]

class EnvState(BaseModel):
    frame: List[List[int]]  # 64x64 grid
    state: str              # "RUNNING", "WIN", "GAME_OVER"
    score: float
    available_actions: List[int]
    step_index: int
    guid: Optional[str] = None
    game_id: Optional[str] = None
    card_id: Optional[str] = None

class MCTSNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None

    # Action taken from parent -> this node
    action: Optional[ActionCandidate] = None

    # Full sequence from root to this node
    action_sequence: List[ActionCandidate] = Field(default_factory=list)

    # Snapshot of environment after applying action_sequence
    state: EnvState

    visits: int = 0
    total_value: float = 0.0
    mean_value: float = 0.0

    is_terminal: bool = False
    depth: int = 0

class MCTSDecision(BaseModel):
    root_state: EnvState
    candidates: List[ActionCandidate]
    children_stats: List[dict]
    best_action: ActionCandidate