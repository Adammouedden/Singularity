from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple

#-----------------------------------#
#-----------------------------------#
#----------- WorldModel ------------#
#-----------------------------------#
#-----------------------------------#

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
    frame: List[List[int]]
    state: str
    score: float
    available_actions: List[int]
    step_index: int
    guid: Optional[str] = None
    game_id: Optional[str] = None
    card_id: Optional[str] = None

#-----------------------------------#
#-----------------------------------#
#-------------- MCTS ---------------#
#-----------------------------------#
#-----------------------------------#

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

    # Tree structure
    children_ids: List[str] = Field(default_factory=list)

    # Cached expansion info
    candidate_actions: List[ActionCandidate] = Field(default_factory=list)
    candidate_scores: List[float] = Field(default_factory=list)
    expanded_action_keys: List[str] = Field(default_factory=list)

class MCTSIteration(BaseModel):
    iteration_index: int
    selected_path: List[str] = Field(default_factory=list)
    expanded_node_id: Optional[str] = None
    candidate_actions: List[ActionCandidate] = Field(default_factory=list)
    simulation_result_value: float = 0.0
    backprop_updates: List[dict] = Field(default_factory=list)

class MCTSDecision(BaseModel):
    root_state: EnvState
    candidates: List[ActionCandidate]
    children_stats: List[dict]
    best_action: ActionCandidate

    root_node_id: Optional[str] = None
    best_child_node_id: Optional[str] = None
    node_registry: Dict[str, MCTSNode] = Field(default_factory=dict)
    iteration_logs: List[MCTSIteration] = Field(default_factory=list)

#-----------------------------------#
#-----------------------------------#
#------- State Abstraction ---------#
#-----------------------------------#
#-----------------------------------#

# Single-frame abstraction
# "What objects and structure exist in this frame?"
class ComponentInfo(BaseModel):
    component_id: str
    value: int
    size: int
    cells: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]   # min_row, min_col, max_row, max_col
    centroid: Tuple[float, float]
    touches_border: bool
    width: int
    height: int


class FrameAbstraction(BaseModel):
    height: int
    width: int
    value_counts: Dict[int, int]
    num_components: int
    components: List[ComponentInfo]


# Transition abstraction
# "What changed from prev frame -> current frame, and was it meaningful?"
class TransitionAbstraction(BaseModel):
    changed_cells: int
    changed_ratio: float
    changed_positions: List[Tuple[int, int]]
    changed_bbox: Optional[Tuple[int, int, int, int]]
    changed_value_pairs: Dict[str, int]  # e.g. "1->0": 4
    border_changed_cells: int
    border_changed_ratio: float
    significant_change: bool