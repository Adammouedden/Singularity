from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionCandidate(BaseModel):
	action: str = Field(
		description="Action identifier (for example: NEXT_STEP, SIMPLIFY, SUBSTITUTE, VERIFY)."
	)
	x: Optional[int] = Field(
		default=None,
		description="Optional positional parameter kept for compatibility with ARC-style actions.",
	)
	y: Optional[int] = Field(
		default=None,
		description="Optional positional parameter kept for compatibility with ARC-style actions.",
	)
	rationale: Optional[str] = Field(
		default=None,
		description="Short explanation for why this action may help.",
	)


class CandidateActions(BaseModel):
	candidates: List[ActionCandidate] = Field(
		description="Top candidate next actions."
	)


class ScoredAction(BaseModel):
	index: int = Field(description="Index of the action in input list")
	score: float = Field(description="Relative quality score for the action")


class EvaluationResult(BaseModel):
	results: List[ScoredAction]


class EnvState(BaseModel):
	frame: Any = Field(
		description="Current problem representation (text, structured object, or grid-like data)."
	)
	state: str = Field(
		description="Episode status (expected: NOT_FINISHED, WIN, or GAME_OVER)."
	)
	score: float = Field(
		description="Numeric progress/reward signal used by search and backup."
	)
	available_actions: List[Any] = Field(
		default_factory=list,
		description="Action identifiers available at this state.",
	)
	step_index: int
	guid: Optional[str] = None
	problem_id: Optional[str] = None
	subproblem_id: Optional[str] = None


class MCTSNode(BaseModel):
	node_id: str
	parent_id: Optional[str] = None

	action: Optional[ActionCandidate] = None
	action_sequence: List[ActionCandidate] = Field(default_factory=list)
	state: EnvState

	visits: int = 0
	total_value: float = 0.0
	mean_value: float = 0.0

	is_terminal: bool = False
	depth: int = 0

	children_ids: List[str] = Field(default_factory=list)

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
