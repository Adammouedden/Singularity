#!/usr/bin/env python3
"""
This is the action proposer.
"""
from schemas.schemas import CandidateActions
from google import genai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

class GeminiWorldModel:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=API_KEY  )
        self.model_name = model_name

        self.action_semantics = {
            "ACTION1": "up",
            "ACTION2": "down",
            "ACTION3": "left",
            "ACTION4": "right",
            "ACTION5": "special/interact/select/rotate/execute",
            "ACTION6": "coordinate action requiring x,y in [0,63]",
            "ACTION7": "undo",
        }

    def _build_prompt(self, frame, actions) -> str:
        # Build a filtered dictionary WITHOUT mutating the original
        valid_actions = {
            f"ACTION{a}": self.action_semantics[f"ACTION{a}"]
            for a in actions
            if f"ACTION{a}" in self.action_semantics
        }
        
        valid_actions_str = "\n".join(
            [f"- {k}: {v}" for k, v in valid_actions.items()]
        )

        prompt = f"""
            You are acting as a barebones world model for an ARC AGI game.

            Valid actions:
            {valid_actions_str}

            Given the current frame, propose the 3 best next actions.

            Rules:
            - Return exactly 3 candidates.
            - Only choose from the valid actions listed above.
            - If you choose ACTION6, include x and y.
            - If action is not ACTION6, x and y should be null.
            - Prefer actions that seem likely to progress the state.
            - Keep rationale short.

            Current frame:
            {frame}
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
            },
        )

        # Depending on SDK version, parsed may be available; fallback to text parse otherwise
        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed

        return CandidateActions.model_validate_json(response.text)