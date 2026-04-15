#!/usr/bin/env python3
"""
This is the critic.
"""
from schemas.schemas import ActionCandidate, EvaluationResult
from typing import List
from google import genai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Will evaluate (state, action), end goal is to evaluate (state, goal-state)
class GeminiCritic:
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
        You are evaluating actions for an ARC AGI task.

        Current grid state:
        {frame}

        Candidate actions (with indices):
        {actions_json}

        Your task:
        Assign a score to EACH action.
        
        Scoring rules:
        - Score range: 0.0 to 1.0
        - 1.0 = very promising
        - 0.0 = very poor

        Heuristics:
        - reward symmetry
        - reward pattern completion
        - reward consistency
        - penalize randomness
        - penalize destructive edits

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
        
    # Evaluate all actions at once so it can compare each one relative to one another
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

        # Handle incorrect structure returned by model
        scores = [0.5] * len(actions)

        for item in parsed.results:
            idx = item.index
            score = max(0.0, min(1.0, item.score))  # normalize it to between 0.0 and 1.0

            if 0 <= idx < len(scores):
                scores[idx] = score

        return scores