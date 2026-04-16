#!/usr/bin/env python3
"""
Transition-aware critic.
"""
from schemas.schemas import ActionCandidate, EvaluationResult
from typing import List
from google import genai
import os
from dotenv import load_dotenv

from arc.state_abstraction import (
    frame_to_key,
    extract_frame_abstraction_from_key,
    extract_transition_abstraction_from_keys,
    summarize_for_llm,
)

load_dotenv(dotenv_path=".env")


API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Will evaluate (state, action), end goal is to evaluate (state, goal-state)
class GeminiCritic:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=API_KEY)
        self.model_name = model_name

        self.frame_summary_cache = {}
        self.next_summary_cache = {}
        self.transition_eval_cache = {}

    def _action_tuple(self, action: ActionCandidate):
        return (action.action, action.x, action.y, action.rationale)

    def _get_prev_summary(self, prev_frame):
        prev_key = frame_to_key(prev_frame)
        if prev_key not in self.frame_summary_cache:
            prev_abs = extract_frame_abstraction_from_key(prev_key, include_cells=False)
            self.frame_summary_cache[prev_key] = summarize_for_llm(prev_abs)
        return self.frame_summary_cache[prev_key]

    def _get_next_summary(self, prev_frame, next_frame):
        prev_key = frame_to_key(prev_frame)
        next_key = frame_to_key(next_frame)
        cache_key = (prev_key, next_key)

        if cache_key not in self.next_summary_cache:
            next_abs = extract_frame_abstraction_from_key(next_key, include_cells=False)
            trans_abs = extract_transition_abstraction_from_keys(prev_key, next_key)
            self.next_summary_cache[cache_key] = summarize_for_llm(next_abs, trans_abs)

        return self.next_summary_cache[cache_key]

    def _serialize_transitions(
        self,
        prev_frame,
        actions: List[ActionCandidate],
        next_frames: List[List[List[int]]],
    ):
        serialized = []
        prev_key = frame_to_key(prev_frame)

        for i, (action, next_frame) in enumerate(zip(actions, next_frames)):
            next_key = frame_to_key(next_frame)
            trans_abs = extract_transition_abstraction_from_keys(prev_key, next_key)

            serialized.append({
                "index": i,
                "action": {
                    "action": action.action,
                    "x": action.x,
                    "y": action.y,
                    "rationale": action.rationale,
                },
                "next_frame_summary": self._get_next_summary(prev_frame, next_frame),
                "transition_summary": {
                    "changed_cells": trans_abs.changed_cells,
                    "changed_ratio": trans_abs.changed_ratio,
                    "changed_bbox": trans_abs.changed_bbox,
                    "changed_value_pairs": trans_abs.changed_value_pairs,
                    "border_changed_cells": trans_abs.border_changed_cells,
                    "border_changed_ratio": trans_abs.border_changed_ratio,
                    "significant_change": trans_abs.significant_change,
                },
            })

        return serialized
    
    def _build_prompt(self, prev_frame, transition_json):
        prev_summary = self._get_prev_summary(prev_frame)

        return f"""
            You are evaluating action transitions for an ARC AGI task.

            Current state abstraction:
            {prev_summary}

            Candidate transitions:
            {transition_json}

            Your task:
            Assign a score to EACH candidate transition.

            Scoring rules:
            - Score range: 0.0 to 1.0
            - 1.0 = very promising transition
            - 0.0 = very poor transition

            Heuristics:
            - reward meaningful change
            - reward pattern completion
            - reward consistency
            - penalize no-op or near-no-op transitions
            - penalize changes mostly confined to border/UI-like regions
            - penalize destructive edits
            - prefer transitions with significant_change=True when justified

            Important:
            - Compare transitions RELATIVE to each other
            - Use a spread of scores
            - Be consistent and deterministic

            Return JSON with:
            - index
            - score
        """

        
    # Evaluate all actions at once so it can compare each one relative to one another
    def evaluate_transitions(
        self,
        prev_frame,
        actions: List[ActionCandidate],
        next_frames: List[List[List[int]]],
    ) -> List[float]:
        if len(actions) != len(next_frames):
            raise ValueError("actions and next_frames must have the same length.")

        prev_key = frame_to_key(prev_frame)
        cache_key = (
            prev_key,
            tuple(self._action_tuple(a) for a in actions),
            tuple(frame_to_key(f) for f in next_frames),
        )

        if cache_key in self.transition_eval_cache:
            return list(self.transition_eval_cache[cache_key])

        transition_json = self._serialize_transitions(prev_frame, actions, next_frames)
        prompt = self._build_prompt(prev_frame, transition_json)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EvaluationResult,
                "temperature": 0.0,
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
            score = max(0.0, min(1.0, item.score))
            if 0 <= idx < len(scores):
                scores[idx] = score

        self.transition_eval_cache[cache_key] = tuple(scores)
        return scores