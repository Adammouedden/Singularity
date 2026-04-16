#!/usr/bin/env python3

from __future__ import annotations

from arc.general_env import GeneralEnv
from evaluation.general_critic import GeneralCritic
from schemas.general_schema import ActionCandidate, EnvState
from search.general_mcts import ShallowMCTS
from world_models.general_world_model import GeneralWorldModel


def _default_initial_problem() -> EnvState:
	return EnvState(
		frame={
			"problem": "Solve for x: 2x + 5 = 17",
			"working": "",
			"final_answer": None,
		},
		state="NOT_FINISHED",
		score=0.0,
		available_actions=[
			"UNDERSTAND",
			"TRANSFORM",
			"SOLVE_SUBPART",
			"VERIFY",
			"FINALIZE",
		],
		step_index=0,
		problem_id="demo-math-001",
	)


def _fallback_action_from_state(state: EnvState) -> ActionCandidate | None:
	if not state.available_actions:
		return None
	return ActionCandidate(
		action=str(state.available_actions[0]),
		rationale="Fallback first available action",
	)


def main() -> None:
	initial_problem = _default_initial_problem()
	env = GeneralEnv(initial_problem=initial_problem)
	proposer = GeneralWorldModel()
	critic = GeneralCritic()
	searcher = ShallowMCTS(
		proposer=proposer,
		critic=critic,
		env=env,
		num_iterations=2,
		exploration_weight=1.4,
	)

	state = env.reset_game()
	committed_actions: list[ActionCandidate] = []

	# Hard cap: commit at most 2 actions, independent of terminal labels.
	for i in range(2):
		action_to_commit: ActionCandidate | None = None

		try:
			decision = searcher.search(state)
			action_to_commit = decision.best_action
		except Exception as e:
			print(f"[warn] MCTS failed at step {i + 1}: {e}")
			action_to_commit = _fallback_action_from_state(state)

		if action_to_commit is None:
			print(f"[warn] No action available to commit at step {i + 1}; stopping.")
			break

		committed_actions.append(action_to_commit)
		state = env.step(state, action_to_commit)

		print(f"step={i + 1} action={action_to_commit.action} state={state.state} score={state.score}")

	print("\nCommitted actions:")
	for idx, action in enumerate(committed_actions, start=1):
		print(f"{idx}. {action.model_dump()}")

	print("\nFinal state:")
	print(state.model_dump())


if __name__ == "__main__":
	main()
