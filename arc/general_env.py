from __future__ import annotations

import os
from typing import Callable, List, Optional

from dotenv import load_dotenv
from google import genai

from schemas.general_schema import ActionCandidate, EnvState

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


class GeneralEnv:
	"""
	Generic environment for replay-based search on non-ARC tasks.

	This environment, not the critic, is responsible for returning terminal
	labels in EnvState.state: NOT_FINISHED, WIN, GAME_OVER.
	"""

	def __init__(
		self,
		initial_problem: EnvState,
		model_name: str = GEMINI_MODEL,
		terminal_evaluator: Optional[Callable[[EnvState], str]] = None,
	):
		self.starting_state = initial_problem.model_copy(deep=True)
		self.client = genai.Client(api_key=API_KEY)
		self.model_name = model_name
		self.terminal_evaluator = terminal_evaluator

	def _normalize_terminal_state(self, state_value: Optional[str]) -> str:
		if not state_value:
			return "NOT_FINISHED"

		normalized = str(state_value).upper().strip()
		if normalized in {"WIN", "GAME_OVER", "NOT_FINISHED"}:
			return normalized
		return "NOT_FINISHED"

	def _apply_terminal_evaluator(self, state: EnvState) -> EnvState:
		if self.terminal_evaluator is None:
			state.state = self._normalize_terminal_state(state.state)
			return state

		evaluated = self._normalize_terminal_state(self.terminal_evaluator(state))
		state.state = evaluated
		return state

	def _build_prompt(self, state: EnvState, action: ActionCandidate) -> str:
		return f"""
		You are simulating one environment transition for a reasoning task
		(for example, a math problem).

		Current EnvState:
		{state.model_dump_json(indent=2)}

		Apply this next action:
		{action.model_dump_json(indent=2)}

		Return ONLY a JSON EnvState for the next step.

		Requirements:
		- state must be one of: NOT_FINISHED, WIN, GAME_OVER
		- score must be a float where larger means closer to a correct final answer
		- available_actions should list valid next actions for the new state
		- increment step_index by 1 from the input state
		- keep problem_id/subproblem_id if present
		- do not add extra keys
		"""

	def reset_game(self) -> EnvState:
		reset_state = self.starting_state.model_copy(deep=True)
		reset_state.step_index = 0
		reset_state.state = self._normalize_terminal_state(reset_state.state)
		return reset_state

	def step(self, state: EnvState, action: ActionCandidate) -> EnvState:
		prompt = self._build_prompt(state, action)

		response = self.client.models.generate_content(
			model=self.model_name,
			contents=prompt,
			config={
				"response_mime_type": "application/json",
				"response_schema": EnvState,
				"temperature": 0.2,
			},
		)

		if hasattr(response, "parsed") and response.parsed is not None:
			next_state = response.parsed
		else:
			next_state = EnvState.model_validate_json(response.text)

		# Keep step progression deterministic even if model output drifts.
		next_state.step_index = state.step_index + 1
		if next_state.problem_id is None:
			next_state.problem_id = state.problem_id
		if next_state.subproblem_id is None:
			next_state.subproblem_id = state.subproblem_id

		return self._apply_terminal_evaluator(next_state)

	def replay_sequence(self, actions: List[ActionCandidate]) -> EnvState:
		state = self.reset_game()

		if state.state in ["WIN", "GAME_OVER"]:
			return state

		for action in actions:
			state = self.step(state=state, action=action)
			if state.state in ["WIN", "GAME_OVER"]:
				break

		return state
