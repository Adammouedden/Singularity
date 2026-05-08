from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import requests
from dotenv import load_dotenv

from schemas.schemas import ActionCandidate, EnvState

load_dotenv(dotenv_path=".env")


class BaseARCEnvironment(ABC):
    def __init__(
        self,
        game_id: Optional[str] = None,
        card_id: Optional[str] = None,
    ):
        self.game_id = game_id
        self.card_id = card_id

    def set_game_context(self, game_id: str, card_id: Optional[str] = None) -> None:
        self.game_id = game_id
        self.card_id = card_id

    def _require_game(self) -> None:
        if not self.game_id:
            raise ValueError("game_id is not set.")

    @abstractmethod
    def reset_game(self) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: ActionCandidate) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def replay_sequence(self, actions: List[ActionCandidate]) -> EnvState:
        raise NotImplementedError


class HTTPARCEnvironment(BaseARCEnvironment):
    def __init__(
        self,
        root_url: str = "https://three.arcprize.org",
        api_key: Optional[str] = None,
        game_id: Optional[str] = None,
        card_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        super().__init__(game_id=game_id, card_id=card_id)
        self.root_url = root_url.rstrip("/")
        self.api_key = api_key or os.getenv("ARC_API_KEY")

        if not self.api_key:
            raise ValueError("Missing ARC_API_KEY in environment or constructor.")

        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            self.session.headers.update({
                "X-API-Key": self.api_key,
                "Accept": "application/json",
            })

    def _require_context(self) -> None:
        self._require_game()
        if not self.card_id:
            raise ValueError("card_id is not set for HTTP mode.")

    def _normalize_frame(self, frame: Any):
        if not frame:
            return frame

        if (
            isinstance(frame, list)
            and isinstance(frame[0], list)
            and len(frame[0]) > 0
            and isinstance(frame[0][0], list)
        ):
            return frame[0]

        return frame

    def _to_env_state(self, game_data: dict, step_index: int) -> EnvState:
        normalized_frame = self._normalize_frame(game_data.get("frame"))

        return EnvState(
            frame=normalized_frame,
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
        self._require_context()

        payload = {
            "game_id": self.game_id,
            "card_id": self.card_id,
        }

        game_data = self._post_cmd("RESET", payload)
        return self._to_env_state(game_data, step_index=0)

    def step(self, state: EnvState, action: ActionCandidate) -> EnvState:
        self._require_context()

        if not state.guid:
            raise ValueError("EnvState.guid is required for HTTP step().")

        payload = {
            "game_id": self.game_id,
            "card_id": self.card_id,
            "guid": state.guid,
            "reasoning": action.rationale,
        }

        if action.action == "ACTION6":
            if action.x is None or action.y is None:
                raise ValueError("ACTION6 requires both x and y.")
            payload["x"] = int(action.x)
            payload["y"] = int(action.y)

        game_data = self._post_cmd(action.action, payload)
        return self._to_env_state(game_data, step_index=state.step_index + 1)

    def replay_sequence(self, actions: List[ActionCandidate]) -> EnvState:
        state = self.reset_game()

        if state.state in ["WIN", "GAME_OVER"]:
            return state

        for action in actions:
            state = self.step(state=state, action=action)
            if state.state in ["WIN", "GAME_OVER"]:
                break

        return state


class OfflineARCEnvironment(BaseARCEnvironment):
    """
    Local toolkit-backed environment.

    Requires:
      pip install arc-agi

    Uses:
      from arc_agi import Arcade, OperationMode
      from arcengine import GameAction
    """

    def __init__(
        self,
        game_id: str,
        render_mode: str = "terminal",
        operation_mode: str = "OFFLINE",
        environments_dir: Optional[str] = None,
        recordings_dir: Optional[str] = None,
        seed: Optional[int] = None,
        DEBUG_SEARCH: bool = False,
    ):
        super().__init__(game_id=game_id, card_id=None)
        self.render_mode = render_mode
        self.seed = seed
        self.DEBUG_SEARCH = DEBUG_SEARCH

        try:
            from arc_agi import Arcade, OperationMode  # type: ignore
            from arcengine import GameAction  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OfflineARCEnvironment requires arc-agi and arcengine. "
                "Install with: pip install arc-agi"
            ) from e

        self.GameAction = GameAction
        self.Arcade = Arcade
        self.OperationMode = OperationMode
        self.environments_dir = environments_dir
        self.recordings_dir = recordings_dir

        if self.DEBUG_SEARCH: print(
            f"[OfflineARCEnvironment.__init__] "
            f"operation_mode={operation_mode}, "
            f"environments_dir={self.environments_dir}"
        )

        mode_enum = getattr(OperationMode, operation_mode.upper())
        self.arc = Arcade(
            operation_mode=mode_enum,
            environments_dir=self.environments_dir,
            recordings_dir=self.recordings_dir,
        )
        self.env = None
        

    def _make_env(self) -> None:
        self._require_game()

        make_kwargs = {
            "render_mode": self.render_mode,
        }
        if self.seed is not None:
            make_kwargs["seed"] = self.seed

        available = self.arc.get_environments()
        available_ids = [g.game_id for g in available]

        if self.DEBUG_SEARCH: print(f"[OfflineARCEnvironment] available local game_ids: {available_ids}")

        self.env = self.arc.make(self.game_id, **make_kwargs)

        if self.env is None:
            raise RuntimeError(
                "OfflineARCEnvironment failed to create env.\n"
                f"Requested game_id: {self.game_id}\n"
                f"Available local games: {available_ids}\n"
                "This usually means OFFLINE mode found no local environment files.\n"
                "Either:\n"
                "1. provide environments_dir pointing to downloaded local games, or\n"
                "2. use toolkit NORMAL/ONLINE mode instead of OFFLINE."
            )

    def _get_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    
    def _convert_nested_arrays(self, obj):
        """
        Recursively convert numpy-like arrays to plain Python lists.
        """
        if hasattr(obj, "tolist"):
            obj = obj.tolist()

        if isinstance(obj, list):
            return [self._convert_nested_arrays(x) for x in obj]

        return obj

    def _normalize_frame(self, frame):
        """
        Normalize frame to plain List[List[int]].

        Handles cases like:
        - numpy arrays
        - [grid]
        - lists containing numpy row arrays
        """
        if frame is None:
            return frame

        frame = self._convert_nested_arrays(frame)

        # unwrap singleton outer list: [grid] -> grid
        if (
            isinstance(frame, list)
            and len(frame) == 1
            and isinstance(frame[0], list)
            and len(frame[0]) > 0
            and isinstance(frame[0][0], list)
        ):
            frame = frame[0]
        
        return frame

    def _extract_available_actions(self) -> List[int]:
        """
        env.action_space is documented as a list of GameAction objects whose .name
        reflects ACTION1/ACTION2/etc. Convert to integer ids [1,2,3,...].
        """
        actions = []
        for a in getattr(self.env, "action_space", []) or []:
            name = getattr(a, "name", "")
            if isinstance(name, str) and name.startswith("ACTION"):
                try:
                    actions.append(int(name.replace("ACTION", "")))
                except ValueError:
                    continue
        return actions

    def _normalize_state_name(self, state_obj: Any) -> str:
        if state_obj is None:
            return "NOT_FINISHED"

        # Enum-like object
        if hasattr(state_obj, "name"):
            return str(state_obj.name)

        s = str(state_obj)
        if "." in s:
            s = s.split(".")[-1]
        return s

    def _to_env_state(self, obs: Any, step_index: int) -> EnvState:
        frame = self._normalize_frame(self._get_attr(obs, "frame"))

        state_obj = self._get_attr(obs, "state", "NOT_FINISHED")
        state = self._normalize_state_name(state_obj)

        score = float(self._get_attr(obs, "score", 0.0))

        # toolkit docs expose env.action_space and note actions update each step
        available_actions = self._extract_available_actions()

        return EnvState(
            frame=frame,
            state=str(state),
            score=score,
            available_actions=available_actions,
            step_index=step_index,
            guid=None,  # offline mode has no API guid
            game_id=self.game_id,
            card_id=None,
        )

    def _to_game_action(self, action_name: str):
        """
        Docs show env.step(GameAction.ACTION1) and note GameAction objects expose .name.
        This maps 'ACTION2' -> GameAction.ACTION2. If your installed version instead
        supports from_name(), you can swap to that.
        """
        try:
            return getattr(self.GameAction, action_name)
        except AttributeError as e:
            raise ValueError(f"Invalid action name for offline mode: {action_name}") from e

    def reset_game(self) -> EnvState:
        if self.env is None:
            self._make_env()
        obs = self.env.reset()
        return self._to_env_state(obs, step_index=0)

    def step(self, state: EnvState, action: ActionCandidate) -> EnvState:
        if self.env is None:
            raise ValueError("Offline environment is not initialized. Call reset_game() first.")

        game_action = self._to_game_action(action.action)

        data = None
        if action.action == "ACTION6":
            if action.x is None or action.y is None:
                raise ValueError("ACTION6 requires both x and y.")
            data = {"x": int(action.x), "y": int(action.y)}

        reasoning = None
        if action.rationale:
            reasoning = {"thought": action.rationale}

        obs = self.env.step(game_action, data=data, reasoning=reasoning)
        return self._to_env_state(obs, step_index=state.step_index + 1)

    def replay_sequence(self, actions: List[ActionCandidate]) -> EnvState:
        state = self.reset_game()

        if state.state in ["WIN", "GAME_OVER"]:
            return state

        for action in actions:
            state = self.step(state=state, action=action)
            if state.state in ["WIN", "GAME_OVER"]:
                break

        return state


def create_environment(
    mode: str,
    game_id: str,
    *,
    card_id: Optional[str] = None,
    root_url: str = "https://three.arcprize.org",
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    render_mode: str = "terminal",
    seed: Optional[int] = None,
    environments_dir: Optional[str] = None,
    recordings_dir: Optional[str] = None,
    DEBUG_SEARCH: bool = False,
) -> BaseARCEnvironment:
    mode = mode.lower()

    if mode == "online":
        return HTTPARCEnvironment(
            root_url=root_url,
            api_key=api_key,
            game_id=game_id,
            card_id=card_id,
            session=session,
        )

    if mode in {"offline", "normal", "online_toolkit"}:
        operation_mode = {
            "offline": "OFFLINE",
            "normal": "NORMAL",
            "online_toolkit": "ONLINE",
        }[mode]

        return OfflineARCEnvironment(
            game_id=game_id,
            render_mode=render_mode,
            operation_mode=operation_mode,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            seed=seed,
            DEBUG_SEARCH=DEBUG_SEARCH,
        )

    raise ValueError(f"Unknown environment mode: {mode}")