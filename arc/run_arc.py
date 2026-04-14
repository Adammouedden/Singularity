#!/usr/bin/env python3

import os
import random
import requests
from dotenv import load_dotenv

from arc.env import ARCEnvironment
from world_models.gemini_world_model import GeminiWorldModel
from evaluation.critic import GeminiCritic
from search.mcts import RootDepthTwoSearch, RootOnlySearch

load_dotenv(dotenv_path=".env")

ROOT_URL = "https://arcprize.org"
API_KEY = os.getenv("ARC_API_KEY")


def get_random_game_id(session: requests.Session) -> str:
    response = session.get(f"{ROOT_URL}/api/games")
    response.raise_for_status()

    games = response.json()
    if not games:
        raise RuntimeError("No games returned from /api/games")

    return random.choice([g["game_id"] for g in games])


def open_scorecard(session: requests.Session) -> str:
    response = session.post(
        f"{ROOT_URL}/api/scorecard/open",
        json={"tags": ["env_test", "root_only_search"]},
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
    if not API_KEY:
        raise ValueError("Missing ARC_API_KEY in ../.env")

    session = requests.Session()
    session.headers.update({
        "X-API-Key": API_KEY,
        "Accept": "application/json",
    })

    game_id = "ls20-9607627b" #get_random_game_id(session)
    card_id = open_scorecard(session)

    print(f"Selected game_id: {game_id}")
    print(f"Opened card_id:   {card_id}")

    env = ARCEnvironment(
        root_url=ROOT_URL,
        api_key=API_KEY,
        game_id=game_id,
        card_id=card_id,
        session=session,
    )

    proposer = GeminiWorldModel()
    critic = GeminiCritic()
    searcher = RootDepthTwoSearch( #RootOnlySearch(
        proposer=proposer,
        critic=critic,
        env=env,
        num_top_root_actions_to_replay=2, #num_top_actions_to_replay=2,
        num_top_child_actions_to_replay=1,
        discount=0.8,
    )

    try:
        # Get current live state
        root_state = env.reset_game()
        print("\n=== ROOT STATE ===")
        print("State:", root_state.state)
        print("Score:", root_state.score)
        print("Available actions:", root_state.available_actions)

        # Run root-only search
        decision = searcher.search(root_state)

        print("\n=== SEARCH DECISION ===")
        print("Best action:", decision.best_action)

        print("\nChildren stats:")
        for i, stat in enumerate(decision.children_stats, start=1):
            print(f"{i}. {stat}")

        # Execute best action in the live episode
        decision.best_action.rationale = f"BEST ACTION TAKEN: {decision.best_action.rationale}"
        next_state = env.step(root_state, decision.best_action)

        print("\n=== AFTER EXECUTING BEST ACTION ===")
        print("State:", next_state.state)
        print("Score:", next_state.score)
        print("Available actions:", next_state.available_actions)

        print("\n=== SEARCH TREE ===")
        for stat in decision.children_stats:
            print(f"root -> {stat['action']['action']} | backed_up_value={stat['backed_up_value']}")
            for gc in stat.get("grandchildren", []):
                print(f"    -> {gc['action']['action']} | value={gc['value']}")
    finally:
        close_scorecard(session, card_id)
        print("Scorecard closed!")
        print(f"\nView results at: {ROOT_URL}/scorecards/{card_id}")


if __name__ == "__main__":
    main()