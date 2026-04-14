#!/usr/bin/env python3

import os
import random
import requests
from dotenv import load_dotenv

from arc.env import ARCEnvironment
from world_models.gemini_world_model import GeminiWorldModel
from evaluation.critic import GeminiCritic
from search.mcts import ShallowMCTS #, RootDepthTwoSearch, RootOnlySearch

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

    game_id = get_random_game_id(session) # "ls20-9607627b" 
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
    searcher = ShallowMCTS(
        proposer=proposer,
        critic=critic,
        env=env,
        num_iterations=6,
        exploration_weight=1.4,
    )

    try:
        # Get current live state
        root_state = env.reset_game()
        
        # Run shallow MCTS
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

        # Execute best action in the live episode
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
        close_scorecard(session, card_id)
        print("Scorecard closed!")
        print(f"\nView results at: {ROOT_URL}/scorecards/{card_id}")


if __name__ == "__main__":
    main()