#!/usr/bin/env python3

import os
import random
import requests
from dotenv import load_dotenv

from arc.env import create_environment
from world_models.gemini_world_model import GeminiWorldModel
from evaluation.critic import GeminiCritic
from search.mcts import ShallowMCTS #, RootDepthTwoSearch, RootOnlySearch
from arc_agi import Arcade, OperationMode

load_dotenv(dotenv_path=".env")

ROOT_URL = "https://three.arcprize.org"
API_KEY = os.getenv("ARC_API_KEY")

# Switch this for development
ENV_MODE = os.getenv("ARC_ENV_MODE", "offline").lower()  # "offline" or "online"
ENVIRONMENTS_DIR = "./arc/environment_files"

def get_random_game_id_http(session: requests.Session) -> str:
    response = session.get(f"{ROOT_URL}/api/games")
    response.raise_for_status()

    games = response.json()
    if not games:
        raise RuntimeError("No games returned from /api/games")

    return random.choice([g["game_id"] for g in games])

def get_random_game_id_offline() -> str:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir="/home/davidorjuela/dev/singularity/arc/environment_files"
    )
    games = arc.get_environments()
    if not games:
        raise RuntimeError("No local games returned from Arcade.get_environments()")

    return random.choice([g.game_id for g in games])

def open_scorecard(session: requests.Session) -> str:
    response = session.post(
        f"{ROOT_URL}/api/scorecard/open",
        json={"tags": ["env_test", "shallow_mcts"]},
    )
    response.raise_for_status()

    data = response.json()
    if "card_id" not in data:
        raise RuntimeError(f"scorecard/open response missing card_id: {data}")

    return data["card_id"]


def close_scorecard(session: requests.Session, card_id: str) -> None:
    response = session.post(
        f"{ROOT_URL}/api/scorecard/close",
        json={"card_id": card_id},
    )
    response.raise_for_status()


def main():
    session = None
    card_id = None

    if ENV_MODE == "online":
        if not API_KEY:
            raise ValueError("Missing ARC_API_KEY for online mode.")

        session = requests.Session()
        session.headers.update({
            "X-API-Key": API_KEY,
            "Accept": "application/json",
        })

        game_id = "ls20-9607627b"  # or get_random_game_id_http(session)
        card_id = open_scorecard(session)

        print(f"Selected game_id: {game_id}")
        print(f"Opened card_id:   {card_id}")

        env = create_environment(
            mode="online",
            game_id=game_id,
            card_id=card_id,
            root_url=ROOT_URL,
            api_key=API_KEY,
            session=session,
        )

    elif ENV_MODE in {"offline", "normal", "online_toolkit"}:
        if ENV_MODE == "offline":
            op_mode = OperationMode.OFFLINE
        elif ENV_MODE == "normal":
            op_mode = OperationMode.NORMAL
        else:
            op_mode = OperationMode.ONLINE

        arc = Arcade(operation_mode=op_mode)
        envs = arc.get_environments()
        env_ids = [g.game_id for g in envs]

        print(f"Toolkit mode: {ENV_MODE}")
        print(f"Available toolkit game_ids: {env_ids[:20]}{'...' if len(env_ids) > 20 else ''}")

        if not env_ids:
            raise RuntimeError(
                f"No environments available in toolkit mode={ENV_MODE}. "
                "If using OFFLINE, you likely have no local environment files."
            )

        # For now, pick a toolkit-visible game ID directly from the list
        game_id = "ls20" if "ls20" in env_ids else env_ids[0]

        print(f"Selected toolkit game_id: {game_id}")

        env = create_environment(
            mode=ENV_MODE,
            game_id=game_id,
            render_mode=None,   #"terminal", # uncomment to render frames in terminal
            seed=0,
            environments_dir=ENVIRONMENTS_DIR,
        )

    else:
        raise ValueError(f"Unsupported ENV_MODE: {ENV_MODE}")

    proposer = GeminiWorldModel()
    critic = GeminiCritic()
    searcher = ShallowMCTS(
        proposer=proposer,
        critic=critic,
        env=env,
        num_iterations=2,
        exploration_weight=1.4,
    )

    try:
        root_state = env.reset_game()
        decision = searcher.search(root_state)

        print("\n=== SEARCH DECISION ===")
        print("Best action:", decision.best_action)
        print("Root node id:", decision.root_node_id)
        print("Best child node id:", decision.best_child_node_id)

        print("\n=== ROOT CHILDREN STATS ===")
        for i, stat in enumerate(decision.children_stats, start=1):
            print(f"{i}. {stat}")

        print("\n=== ITERATION LOGS ===")
        for log in decision.iteration_logs:
            print(f"Iteration {log.iteration_index}")
            print("  Selected path:", log.selected_path)
            print("  Expanded node:", log.expanded_node_id)
            print("  Candidate actions:", [a.action for a in log.candidate_actions])
            print("  Simulation value:", log.simulation_result_value)
            print("  Backprop:", log.backprop_updates)

        decision.best_action.rationale = f"BEST ACTION TAKEN: {decision.best_action.rationale}"
        next_state = env.step(root_state, decision.best_action)

        print("\n=== AFTER EXECUTING BEST ACTION ===")
        print("State:", next_state.state)
        print("Score:", next_state.score)
        print("Available actions:", next_state.available_actions)

        print("\n=== ROOT TREE SNAPSHOT ===")
        for stat in decision.children_stats:
            print(
                f"root -> {stat['action']['action']} "
                f"| node={stat['node_id']} "
                f"| visits={stat['visits']} "
                f"| mean_value={stat['mean_value']}"
            )

    finally:
        if ENV_MODE == "online" and session is not None and card_id is not None:
            close_scorecard(session, card_id)
            print("Scorecard closed!")
            print(f"\nView results at: https://arcprize.org/scorecards/{card_id}")


if __name__ == "__main__":
    main()