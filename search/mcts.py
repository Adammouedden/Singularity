from __future__ import annotations

import uuid
from typing import List, Dict, Any, Tuple

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
    

#-----------------------------------#
#-----------------------------------#
#------ ROOT DEPTH TWO SEARCH ------#
#-----------------------------------#
#-----------------------------------#


class RootDepthTwoSearch:
    def __init__(
        self,
        proposer,
        critic,
        env,
        num_top_root_actions_to_replay: int = 2,
        num_top_child_actions_to_replay: int = 1,
        discount: float = 0.8,
    ):
        """
        proposer: object with propose_actions(frame, actions) -> CandidateActions
        critic: object with evaluate_batch(frame, actions) -> List[float]
        env: ARCEnvironment with replay_sequence(actions) -> EnvState
        """
        self.proposer = proposer
        self.critic = critic
        self.env = env
        self.num_top_root_actions_to_replay = num_top_root_actions_to_replay
        self.num_top_child_actions_to_replay = num_top_child_actions_to_replay
        self.discount = discount

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

    def search(self, root_state: EnvState) -> MCTSDecision:
        """
        Depth-2 replay-based search:

        root
          -> top root children
               -> best child action from each replayed root child

        Returns the best root action.
        """
        root = self._make_root_node(root_state)

        # ----- Root expansion -----
        root_ranked = self._rank_actions_for_state(root_state)
        if not root_ranked:
            raise ValueError("Proposer returned no candidate actions at root.")

        root_candidates = [action for action, _ in root_ranked]
        top_root_ranked = root_ranked[: self.num_top_root_actions_to_replay]

        root_child_nodes: List[MCTSNode] = []
        children_stats: List[Dict[str, Any]] = []

        for root_action, root_critic_score in top_root_ranked:
            # Materialize the root child by replaying [root_action]
            root_child_state = self.env.replay_sequence([root_action])

            root_immediate_value = self._score_materialized_state(
                parent_state=root_state,
                child_state=root_child_state,
                critic_score=root_critic_score,
            )

            root_child_node = self._make_child_node(
                parent=root,
                action=root_action,
                child_state=root_child_state,
                value=root_immediate_value,
            )

            branch_stat: Dict[str, Any] = {
                "action": root_action.model_dump(),
                "critic_score": root_critic_score,
                "env_state": root_child_state.state,
                "env_score": root_child_state.score,
                "immediate_value": root_immediate_value,
                "visits": root_child_node.visits,
                "total_value": root_child_node.total_value,
                "mean_value": root_child_node.mean_value,
                "best_grandchild_action": None,
                "best_grandchild_value": None,
                "backed_up_value": root_immediate_value,
                "grandchildren": [],
            }

            # ----- Depth 2 expansion -----
            if root_child_state.state not in ["WIN", "GAME_OVER"]:
                child_ranked = self._rank_actions_for_state(root_child_state)

                if child_ranked:
                    top_child_ranked = child_ranked[: self.num_top_child_actions_to_replay]

                    best_grandchild_value = None
                    best_grandchild_action = None

                    for child_action, child_critic_score in top_child_ranked:
                        grandchild_sequence = [root_action, child_action]
                        grandchild_state = self.env.replay_sequence(grandchild_sequence)

                        grandchild_value = self._score_materialized_state(
                            parent_state=root_child_state,
                            child_state=grandchild_state,
                            critic_score=child_critic_score,
                        )

                        branch_stat["grandchildren"].append({
                            "action": child_action.model_dump(),
                            "critic_score": child_critic_score,
                            "env_state": grandchild_state.state,
                            "env_score": grandchild_state.score,
                            "value": grandchild_value,
                        })

                        if (
                            best_grandchild_value is None
                            or grandchild_value > best_grandchild_value
                        ):
                            best_grandchild_value = grandchild_value
                            best_grandchild_action = child_action

                    if best_grandchild_value is not None:
                        backed_up_value = root_immediate_value + self.discount * best_grandchild_value
                        root_child_node.total_value = backed_up_value
                        root_child_node.mean_value = backed_up_value

                        branch_stat["best_grandchild_action"] = (
                            best_grandchild_action.model_dump()
                            if best_grandchild_action is not None
                            else None
                        )
                        branch_stat["best_grandchild_value"] = best_grandchild_value
                        branch_stat["backed_up_value"] = backed_up_value

            root_child_nodes.append(root_child_node)
            children_stats.append(branch_stat)

        if not root_child_nodes:
            best_action = root_ranked[0][0]
        else:
            best_child = max(root_child_nodes, key=lambda node: node.mean_value)
            if best_child.action is None:
                raise ValueError("Best child action is None.")
            best_action = best_child.action

        return MCTSDecision(
            root_state=root_state,
            candidates=root_candidates,
            children_stats=children_stats,
            best_action=best_action,
        )