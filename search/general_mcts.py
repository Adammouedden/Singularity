from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

from schemas.general_schema import (
	ActionCandidate,
	EnvState,
	MCTSDecision,
	MCTSIteration,
	MCTSNode,
)


class ShallowMCTS:
	def __init__(
		self,
		proposer,
		critic,
		env,
		num_iterations: int = 6,
		exploration_weight: float = 1.4,
	):
		self.proposer = proposer
		self.critic = critic
		self.env = env
		self.num_iterations = num_iterations
		self.exploration_weight = exploration_weight

		self.node_registry: Dict[str, MCTSNode] = {}
		self.iteration_logs: List[MCTSIteration] = []

	def _action_key(self, action: ActionCandidate) -> str:
		return f"{action.action}|{action.x}|{action.y}"

	def _make_root_node(self, root_state: EnvState) -> MCTSNode:
		root = MCTSNode(
			node_id=str(uuid.uuid4()),
			parent_id=None,
			action=None,
			action_sequence=[],
			state=root_state,
			visits=0,
			total_value=0.0,
			mean_value=0.0,
			is_terminal=root_state.state in ["WIN", "GAME_OVER"],
			depth=0,
		)
		self.node_registry[root.node_id] = root
		return root

	def _make_child_node(
		self,
		parent: MCTSNode,
		action: ActionCandidate,
		child_state: EnvState,
	) -> MCTSNode:
		child = MCTSNode(
			node_id=str(uuid.uuid4()),
			parent_id=parent.node_id,
			action=action,
			action_sequence=parent.action_sequence + [action],
			state=child_state,
			visits=0,
			total_value=0.0,
			mean_value=0.0,
			is_terminal=child_state.state in ["WIN", "GAME_OVER"],
			depth=parent.depth + 1,
		)
		self.node_registry[child.node_id] = child
		parent.children_ids.append(child.node_id)
		return child

	def _rank_actions_for_state(
		self,
		state: EnvState,
	) -> List[Tuple[ActionCandidate, float]]:
		proposal_result = self.proposer.propose_actions(
			frame=state.frame,
			actions=state.available_actions,
		)
		candidates: List[ActionCandidate] = proposal_result.candidates

		if not candidates:
			return []

		scores = self.critic.evaluate_batch(state.frame, candidates)

		ranked = list(zip(candidates, scores))
		ranked.sort(key=lambda x: x[1], reverse=True)
		return ranked

	def _ensure_candidates(self, node: MCTSNode) -> None:
		if node.is_terminal:
			return

		if node.candidate_actions:
			return

		ranked = self._rank_actions_for_state(node.state)

		deduped_actions = []
		deduped_scores = []
		seen_keys = set()

		for action, score in ranked:
			key = self._action_key(action)
			if key in seen_keys:
				continue
			seen_keys.add(key)
			deduped_actions.append(action)
			deduped_scores.append(score)

		node.candidate_actions = deduped_actions
		node.candidate_scores = deduped_scores

	def _has_unexpanded_actions(self, node: MCTSNode) -> bool:
		self._ensure_candidates(node)
		expanded_keys = set(node.expanded_action_keys)

		for action in node.candidate_actions:
			if self._action_key(action) not in expanded_keys:
				return True
		return False

	def _best_child_ucb(self, node: MCTSNode) -> MCTSNode:
		if not node.children_ids:
			raise ValueError("Cannot compute UCB on a node with no children.")

		best_score = float("-inf")
		best_child: Optional[MCTSNode] = None

		parent_visits = max(node.visits, 1)

		for child_id in node.children_ids:
			child = self.node_registry[child_id]

			if child.visits == 0:
				ucb_score = float("inf")
			else:
				exploitation = child.mean_value
				exploration = self.exploration_weight * math.sqrt(
					math.log(parent_visits + 1) / child.visits
				)
				ucb_score = exploitation + exploration

			if ucb_score > best_score:
				best_score = ucb_score
				best_child = child

		if best_child is None:
			raise ValueError("Failed to select a best child by UCB.")

		return best_child

	def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[str]]:
		current = root
		selected_path = [current.node_id]

		while True:
			if current.is_terminal:
				return current, selected_path

			self._ensure_candidates(current)

			if self._has_unexpanded_actions(current):
				return current, selected_path

			if not current.children_ids:
				return current, selected_path

			current = self._best_child_ucb(current)
			selected_path.append(current.node_id)

	def _score_materialized_state(
		self,
		parent_state: EnvState,
		child_state: EnvState,
		critic_score: float,
	) -> float:
		terminal_bonus = 0.0
		if child_state.state == "WIN":
			terminal_bonus = 1.0
		elif child_state.state == "GAME_OVER":
			terminal_bonus = -1.0

		env_bonus = 0.0
		if child_state.score > parent_state.score:
			env_bonus = 0.1
		elif child_state.score < parent_state.score:
			env_bonus = -0.05

		return critic_score + terminal_bonus + env_bonus

	def _rollout(
		self,
		start_state: EnvState,
		base_action_sequence: List[ActionCandidate],
		max_steps: int = 2,
	) -> EnvState:
		current_state = start_state
		current_sequence = list(base_action_sequence)

		for rollout_step in range(max_steps):
			if current_state.state in ["WIN", "GAME_OVER"]:
				break

			ranked = self._rank_actions_for_state(current_state)
			if not ranked:
				break

			action = ranked[0][0]
			current_sequence.append(action)

			current_state = self.env.replay_sequence(current_sequence)
			rationale = (action.rationale or "").replace("\n", " ").strip()
			print(
				f"[SIMULATED][NOT_COMMITTED] rollout_step={rollout_step + 1} "
				f"action={action.action} state={current_state.state} score={current_state.score} "
				f"rationale='{rationale}'"
			)

		return current_state

	def _expand(self, node: MCTSNode) -> Tuple[MCTSNode, float]:
		self._ensure_candidates(node)

		if not self._has_unexpanded_actions(node):
			raise ValueError(
				f"Tried to expand a fully expanded node. "
				f"node_id={node.node_id}, "
				f"candidate_keys={[self._action_key(a) for a in node.candidate_actions]}, "
				f"expanded_keys={node.expanded_action_keys}"
			)

		chosen_action = None
		chosen_score = None

		for action, score in zip(node.candidate_actions, node.candidate_scores):
			key = self._action_key(action)
			if key not in node.expanded_action_keys:
				chosen_action = action
				chosen_score = score
				node.expanded_action_keys.append(key)
				break

		if chosen_action is None or chosen_score is None:
			raise ValueError(
				f"Failed to find an unexpanded action. "
				f"node_id={node.node_id}, "
				f"candidate_keys={[self._action_key(a) for a in node.candidate_actions]}, "
				f"expanded_keys={node.expanded_action_keys}"
			)

		child_sequence = node.action_sequence + [chosen_action]
		child_state = self.env.replay_sequence(child_sequence)
		rationale = (chosen_action.rationale or "").replace("\n", " ").strip()
		print(
			f"[SIMULATED][NOT_COMMITTED] expand_depth={node.depth + 1} "
			f"action={chosen_action.action} state={child_state.state} score={child_state.score} "
			f"rationale='{rationale}'"
		)

		child_node = self._make_child_node(
			parent=node,
			action=chosen_action,
			child_state=child_state,
		)

		rollout_state = self._rollout(
			start_state=child_state,
			base_action_sequence=child_sequence,
			max_steps=2,
		)

		leaf_value = self._score_materialized_state(
			parent_state=node.state,
			child_state=rollout_state,
			critic_score=chosen_score,
		)

		return child_node, leaf_value

	def _evaluate_terminal_or_stuck_node(self, node: MCTSNode) -> float:
		if node.state.state == "WIN":
			return 1.0
		if node.state.state == "GAME_OVER":
			return -1.0
		return node.mean_value if node.visits > 0 else 0.0

	def _backpropagate(self, path_node_ids: List[str], value: float) -> List[dict]:
		updates: List[dict] = []

		for node_id in reversed(path_node_ids):
			node = self.node_registry[node_id]
			node.visits += 1
			node.total_value += value
			node.mean_value = node.total_value / node.visits

			updates.append({
				"node_id": node.node_id,
				"visits": node.visits,
				"total_value": node.total_value,
				"mean_value": node.mean_value,
			})

		return updates

	def _build_root_children_stats(self, root: MCTSNode) -> List[dict]:
		stats: List[dict] = []

		for child_id in root.children_ids:
			child = self.node_registry[child_id]
			stats.append({
				"node_id": child.node_id,
				"action": child.action.model_dump() if child.action else None,
				"env_state": child.state.state,
				"env_score": child.state.score,
				"visits": child.visits,
				"total_value": child.total_value,
				"mean_value": child.mean_value,
				"depth": child.depth,
				"children_ids": child.children_ids,
			})

		stats.sort(key=lambda x: (x["visits"], x["mean_value"]), reverse=True)
		return stats

	def search(self, root_state: EnvState) -> MCTSDecision:
		self.node_registry = {}
		self.iteration_logs = []

		root = self._make_root_node(root_state)
		self._ensure_candidates(root)

		for iteration_index in range(1, self.num_iterations + 1):
			selected_node, selected_path = self._select(root)

			iteration_log = MCTSIteration(
				iteration_index=iteration_index,
				selected_path=selected_path.copy(),
				candidate_actions=selected_node.candidate_actions.copy(),
			)

			if selected_node.is_terminal:
				leaf_value = self._evaluate_terminal_or_stuck_node(selected_node)
				iteration_log.expanded_node_id = selected_node.node_id
				iteration_log.simulation_result_value = leaf_value
				iteration_log.backprop_updates = self._backpropagate(selected_path, leaf_value)
			else:
				child_node, leaf_value = self._expand(selected_node)

				full_backprop_path = selected_path + [child_node.node_id]

				iteration_log.expanded_node_id = child_node.node_id
				iteration_log.simulation_result_value = leaf_value
				iteration_log.backprop_updates = self._backpropagate(full_backprop_path, leaf_value)

			self.iteration_logs.append(iteration_log)

		root_children_stats = self._build_root_children_stats(root)

		if root.children_ids:
			best_child = max(
				(self.node_registry[child_id] for child_id in root.children_ids),
				key=lambda node: (node.visits, node.mean_value),
			)
			if best_child.action is None:
				raise ValueError("Best child action is None.")
			best_action = best_child.action
			best_child_node_id = best_child.node_id
		else:
			if not root.candidate_actions:
				raise ValueError("No root candidates available.")
			best_action = root.candidate_actions[0]
			best_child_node_id = None

		return MCTSDecision(
			root_state=root_state,
			candidates=root.candidate_actions,
			children_stats=root_children_stats,
			best_action=best_action,
			root_node_id=root.node_id,
			best_child_node_id=best_child_node_id,
			node_registry=self.node_registry.copy(),
			iteration_logs=self.iteration_logs.copy(),
		)
