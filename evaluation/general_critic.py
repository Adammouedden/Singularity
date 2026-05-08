#!/usr/bin/env python3
"""
General critic for non-ARC tasks (for example, math/problem-solving).
"""

import os
from typing import List

from dotenv import load_dotenv
from google import genai

from schemas.general_schema import ActionCandidate, EvaluationResult

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


class GeneralCritic:
	def __init__(self, model_name: str = GEMINI_MODEL):
		self.client = genai.Client(api_key=API_KEY)
		self.model_name = model_name

	def _serialize_actions(self, actions: List[ActionCandidate]):
		return [
			{
				"action": a.action,
				"x": a.x,
				"y": a.y,
				"rationale": a.rationale,
			}
			for a in actions
		]

	def _build_prompt(self, frame, actions_json):
		return f"""
		You are evaluating next-step actions for a general reasoning task
		(for example, solving a math problem).

		Current state:
		{frame}

		Candidate actions (with indices):
		{actions_json}

		Your task:
		Assign a score to EACH action.

		Scoring rules:
		- Score range: 0.0 to 1.0
		- 1.0 = very promising next step
		- 0.0 = very poor next step

		Heuristics:
		- reward logical progress toward a correct final solution
		- reward consistency with known constraints
		- reward steps that reduce ambiguity or simplify the problem
		- penalize unjustified leaps
		- penalize contradictory or irrelevant steps

		Important:
		- Compare actions RELATIVE to each other
		- Use a spread of scores (avoid all similar)
		- Be consistent and deterministic

		Output format:
		Return JSON with:
		- index (same as input order)
		- score

		Example:
		[
		{{ "index": 0, "score": 0.7 }},
		{{ "index": 1, "score": 0.4 }}
		]
		"""

	def evaluate_batch(self, frame, actions: List[ActionCandidate]) -> List[float]:
		actions_json = self._serialize_actions(actions)
		prompt = self._build_prompt(frame, actions_json)

		response = self.client.models.generate_content(
			model=self.model_name,
			contents=prompt,
			config={
				"response_mime_type": "application/json",
				"response_schema": EvaluationResult,
				"temperature": 0.2,
			},
		)

		if hasattr(response, "parsed") and response.parsed:
			parsed = response.parsed
		else:
			parsed = EvaluationResult.model_validate_json(response.text)

		scores = [0.5] * len(actions)

		for item in parsed.results:
			idx = item.index
			score = max(0.0, min(1.0, item.score))

			if 0 <= idx < len(scores):
				scores[idx] = score

		return scores
