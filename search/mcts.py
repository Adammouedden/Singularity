from __future__ import annotations

import uuid
from typing import List, Dict, Any

from schemas.schemas import (
    ActionCandidate,
    EnvState,
    MCTSDecision,
    MCTSNode
)


class RootOnlySearch:
    def __init__(self, proposer, critic, env, num_top_actions_to_replay: int = 2):
        """
        proposer: object with propose_actions(frame, available_actions) -> CandidateActions
        critic: object with evaluate_batch(frame, actions) -> List[float]
        env: ARCEnvironment with replay_sequence(actions) -> EnvState
        """
        self.proposer = proposer
        self.critic = critic
        self.env = env
        self.num_top_actions_to_replay = num_top_actions_to_replay

    def _make_root_node(self, root_state: EnvState) -> MCTSNode:
        return MCTSNode(
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

    def _make_child_node(
        self,
        parent: MCTSNode,
        action: ActionCandidate,
        child_state: EnvState,
        value: float,
    ) -> MCTSNode:
        return MCTSNode(
            node_id=str(uuid.uuid4()),
            parent_id=parent.node_id,
            action=action,
            action_sequence=parent.action_sequence + [action],
            state=child_state,
            visits=1,
            total_value=value,
            mean_value=value,
            is_terminal=child_state.state in ["WIN", "GAME_OVER"],
            depth=parent.depth + 1,
        )

    def search(self, root_state: EnvState) -> MCTSDecision:
        """
        Root-only search:
        1. proposer proposes candidate actions at root
        2. critic scores them
        3. replay top-K actions
        4. choose best replayed action
        """
        root = self._make_root_node(root_state)

        # 1. Propose actions from current root state
        proposal_result = self.proposer.propose_actions(
            frame=root_state.frame,
            actions=root_state.available_actions,
        )
        candidates: List[ActionCandidate] = proposal_result.candidates

        if not candidates:
            raise ValueError("Proposer returned no candidate actions.")

        # 2. Critic scores all candidates relative to each other
        scores = self.critic.evaluate_batch(root_state.frame, candidates)

        ranked = list(zip(candidates, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        # 3. Replay top-K only
        top_ranked = ranked[: self.num_top_actions_to_replay]

        child_nodes: List[MCTSNode] = []
        children_stats: List[Dict[str, Any]] = []

        for action, critic_score in top_ranked:
            child_sequence = [action]
            child_state = self.env.replay_sequence(child_sequence)

            terminal_bonus = 0.0
            if child_state.state == "WIN":
                terminal_bonus = 1.0
            elif child_state.state == "GAME_OVER":
                terminal_bonus = -1.0

            env_bonus = 0.0
            if child_state.score > root_state.score:
                env_bonus = 0.1

            child_value = critic_score + terminal_bonus + env_bonus

            child_node = self._make_child_node(
                parent=root,
                action=action,
                child_state=child_state,
                value=child_value,
            )
            
            child_nodes.append(child_node)

            children_stats.append({
                "action": action.model_dump(),
                "critic_score": critic_score,
                "env_state": child_state.state,
                "env_score": child_state.score,
                "visits": child_node.visits,
                "total_value": child_node.total_value,
                "mean_value": child_node.mean_value,
            })

        if not child_nodes:
            # fallback: choose top critic-ranked action even if replay phase failed/skipped
            best_action = ranked[0][0]
        else:
            # 4. choose best replayed child by mean_value
            best_child = max(child_nodes, key=lambda node: node.mean_value)
            best_action = best_child.action

        if best_action is None:
            raise ValueError("Best action could not be determined.")

        return MCTSDecision(
            root_state=root_state,
            candidates=candidates,
            children_stats=children_stats,
            best_action=best_action,
        )