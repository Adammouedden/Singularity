from __future__ import annotations

import os
import requests
from typing import List, Optional
from dotenv import load_dotenv

from schemas.schemas import ActionCandidate, EnvState

load_dotenv(dotenv_path=".env")


class ARCEnvironment:
    def __init__(
        self,
        root_url: str = "https://arcprize.org",
        api_key: Optional[str] = None,
        game_id: Optional[str] = None,
        card_id: Optional[str] = None,
    ):
        self.root_url = root_url.rstrip("/")
        self.api_key = api_key or os.getenv("ARC_API_KEY")
        self.game_id = game_id
        self.card_id = card_id

        if not self.api_key:
            raise ValueError("Missing ARC_API_KEY in environment or constructor.")

        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        })

    def set_game_context(self, game_id: str, card_id: str) -> None:
        self.game_id = game_id
        self.card_id = card_id

    def _require_context(self) -> None:
        if not self.game_id:
            raise ValueError("game_id is not set.")
        if not self.card_id:
            raise ValueError("card_id is not set.")

    def _to_env_state(self, game_data: dict, step_index: int) -> EnvState:
        return EnvState(
            frame=game_data["frame"],
            state=game_data["state"],
            score=float(game_data.get("score", 0.0)),
            available_actions=game_data.get("available_actions", []),
            step_index=step_index,
            guid=game_data.get("guid"),
            game_id=self.game_id,
            card_id=self.card_id,
        )

    def _post_cmd(self, action_name: str, payload: dict) -> dict:
        url = f"{self.root_url}/api/cmd/{action_name}"
        response = self.session.post(url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"ARC command failed.\n"
                f"Action: {action_name}\n"
                f"Status: {response.status_code}\n"
                f"Body: {response.text}"
            )

        return response.json()

    def reset_game(self) -> EnvState:
        """
        Starts or restarts the current game with RESET.
        Returns:
            EnvState
        """
        self._require_context()

        payload = {
            "game_id": self.game_id,
            "card_id": self.card_id,
        }

        game_data = self._post_cmd("RESET", payload)
        env_state = self._to_env_state(game_data, step_index=0)
        return env_state

    def step(self, state: EnvState, action: ActionCandidate) -> EnvState:
        """
        Apply one action to the current live ARC state.
        Returns:
            next EnvState
        """
        self._require_context()

        if not state.guid:
            raise ValueError("EnvState.guid is required for step().")

        payload = {
            "game_id": self.game_id,
            "card_id": self.card_id,
            "guid": state.guid,
        }

        action_name = action.action

        if action_name == "ACTION6":
            if action.x is None or action.y is None:
                raise ValueError("ACTION6 requires both x and y.")
            payload["x"] = int(action.x)
            payload["y"] = int(action.y)

        game_data = self._post_cmd(action_name, payload)
        next_state = self._to_env_state(game_data, step_index=state.step_index + 1)
        return next_state

    def replay_sequence(self, actions: List[ActionCandidate]) -> EnvState:
        """
        Reconstruct a node state by:
        1. RESETting the game
        2. replaying the given action sequence in order

        Returns:
            final EnvState
        """
        state = self.reset_game()

        if state.state in ["WIN", "GAME_OVER"]:
            return state

        for action in actions:
            state = self.step(state=state, action=action)

            if state.state in ["WIN", "GAME_OVER"]:
                break

        return state