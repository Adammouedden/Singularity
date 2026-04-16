#!/usr/bin/env python3
"""
This is the action proposer.
"""
from schemas.schemas import CandidateActions
from google import genai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from arc.state_abstraction import (
    extract_frame_abstraction_from_key,
    frame_to_key,
    summarize_for_llm,
)

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

class GeminiWorldModel:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=API_KEY  )
        self.model_name = model_name

        self.frame_summary_cache = {}
        self.proposal_cache = {}

        self.action_semantics = {
            "ACTION1": "up",
            "ACTION2": "down",
            "ACTION3": "left",
            "ACTION4": "right",
            "ACTION5": "special/interact/select/rotate/execute",
            "ACTION6": "coordinate action requiring x,y in [0,63]",
            "ACTION7": "undo",
        }
    
    def _get_frame_summary(self, frame):
        frame_key = frame_to_key(frame)
        if frame_key not in self.frame_summary_cache:
            frame_abs = extract_frame_abstraction_from_key(frame_key, include_cells=False)
            self.frame_summary_cache[frame_key] = summarize_for_llm(frame_abs)
        return self.frame_summary_cache[frame_key]

    def _build_prompt(self, frame, actions) -> str:
        frame_summary = self._get_frame_summary(frame)

        valid_actions = {
            f"ACTION{a}": self.action_semantics[f"ACTION{a}"]
            for a in actions
            if f"ACTION{a}" in self.action_semantics
        }

        valid_actions_str = "\n".join([f"- {k}: {v}" for k, v in valid_actions.items()])

        prompt = f"""
            You are acting as a barebones world model for an ARC AGI game.

            Valid actions:
            {valid_actions_str}

            {frame_summary}

            Given the current frame, propose the 3 best next actions.

            Rules:
            - Return exactly 3 candidates.
            - Only choose from the valid actions listed above.
            - If you choose ACTION6, include x and y.
            - If action is not ACTION6, x and y should be null.
            - Prefer actions that seem likely to progress the state.
            - Keep rationale short.

            Current raw frame:
            {frame}
        """
        return prompt

    def propose_actions(self, frame, actions) -> CandidateActions:
        cache_key = (frame_to_key(frame), tuple(actions))
        if cache_key in self.proposal_cache:
            return self.proposal_cache[cache_key].model_copy(deep=True)

        prompt = self._build_prompt(frame, actions)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CandidateActions,
            },
        )

        if hasattr(response, "parsed") and response.parsed is not None:
            result = response.parsed
        else:
            result = CandidateActions.model_validate_json(response.text)

        self.proposal_cache[cache_key] = result.model_copy(deep=True)
        return result