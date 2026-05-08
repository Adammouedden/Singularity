#!/usr/bin/env python3
"""
General action proposer for non-ARC tasks (for example, math/problem-solving).
"""

import os

from dotenv import load_dotenv
from google import genai

from schemas.general_schema import CandidateActions

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


class GeneralWorldModel:
	def __init__(self, model_name: str = GEMINI_MODEL):
		self.client = genai.Client(api_key=API_KEY)
		self.model_name = model_name
		# use basic simple actions so lLLM can more freely interpret an action
		# the actual reasoning will be store in rationale
		self.universal_actions = [
			"UNDERSTAND",
			"TRANSFORM",
			"SOLVE_SUBPART",
			"VERIFY",
			"FINALIZE",
		]

	def _build_prompt(self, frame, actions) -> str:
		provided_actions = [str(a) for a in actions] if actions else []
		valid_actions = provided_actions if provided_actions else self.universal_actions
		valid_actions_str = "\n".join([f"- {a}" for a in valid_actions])

		prompt = f"""
			You are acting as a world model for a general problem-solving task
			(for example, a math problem).

			Current problem state:
			{frame}

			Valid actions:
			{valid_actions_str if valid_actions_str else '- (none provided)'}

			Task:
			Propose exactly 3 promising next actions.

			Rules:
			- Return exactly 3 candidates.
			- If valid actions are provided, only use those.
			- Keep the action label high-level (from the small universal action vocabulary).
			- Put the detailed mathematical operation in rationale.
			- Do NOT invent fine-grained action names like APPLY_LOG_BOTH_SIDES.
			- Use concise rationale for each candidate.
			- Keep x and y as null unless the action truly needs coordinates.
			- Prefer actions that make measurable progress toward solving the problem.
			- Include at least one conservative/safe action when uncertain.
		"""
		return prompt

	def propose_actions(self, frame, actions) -> CandidateActions:
		prompt = self._build_prompt(frame, actions)

		response = self.client.models.generate_content(
			model=self.model_name,
			contents=prompt,
			config={
				"response_mime_type": "application/json",
				"response_schema": CandidateActions,
				"temperature": 0.2,
			},
		)

		if hasattr(response, "parsed") and response.parsed is not None:
			return response.parsed

		return CandidateActions.model_validate_json(response.text)
