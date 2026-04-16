#!/usr/bin/env python3

import os
import random
import requests
from dotenv import load_dotenv

from arc.env import create_environment
from world_models.gemini_world_model import GeminiWorldModel
from world_models.CNN_world_model import CNNWorldModel
from evaluation.critic import GeminiCritic
from search.mcts import ShallowMCTS #, RootDepthTwoSearch, RootOnlySearch
from arc_agi import Arcade, OperationMode

from arc.state_abstraction import (
    extract_frame_abstraction,
    extract_transition_abstraction,
    summarize_for_llm,
)

load_dotenv(dotenv_path=".env")

ROOT_URL = "https://three.arcprize.org"
API_KEY = os.getenv("ARC_API_KEY")

# Switch this for development
ENV_MODE = os.getenv("ARC_ENV_MODE", "offline").lower()  # "offline" or "online"
ENVIRONMENTS_DIR = "./arc/environment_files"
DEBUG_SEARCH = False # Noisy debugs for full MCTS detail

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
            DEBUG_SEARCH=DEBUG_SEARCH
        )

    elif ENV_MODE in {"offline", "normal", "online_toolkit"}:
        if ENV_MODE == "offline":
            op_mode = OperationMode.OFFLINE
        elif ENV_MODE == "normal":
            op_mode = OperationMode.NORMAL
        else:
            op_mode = OperationMode.ONLINE

        arc = Arcade(
            operation_mode=op_mode,
            environments_dir=ENVIRONMENTS_DIR,
        )
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
            render_mode="terminal",   #"None", # uncomment to render frames in terminal
            seed=0,
            environments_dir=ENVIRONMENTS_DIR,
        )

    else:
        raise ValueError(f"Unsupported ENV_MODE: {ENV_MODE}")

    proposer = GeminiWorldModel() #CNNWorldModel(env.game_id)
    CNN_as_proposer = isinstance(proposer, CNNWorldModel) 

    critic = GeminiCritic()
    
    searcher = ShallowMCTS(
        proposer=proposer,
        critic=critic,
        CNN_as_proposer=CNN_as_proposer,
        env=env,
        num_iterations=6, # Phase B: = 2,, comparing root children ranking (proposer + critic intelligence)
        exploration_weight=1.0,
        rollout_depth=2,
    )

    best_action_history = []
    episode_decisions = []
    
    try:
        state = env.reset_game()
        step_counter = 0
        max_real_steps = 40

        while state.state not in ["WIN", "GAME_OVER"] and step_counter < max_real_steps:
            print(f"\n=== REAL STEP {step_counter} ===")
            print("State:", state.state)
            print("Score:", state.score)
            print("Available actions:", state.available_actions)

            frame_abs = extract_frame_abstraction(state.frame, include_cells=True)
            print("\n=== FRAME ABSTRACTION ===")
            print(summarize_for_llm(frame_abs))
            #print(state.frame)
            decision = searcher.search(state)

            episode_decisions.append(decision)
            best_action_history.append(decision.best_action)

            print("\n=== SEARCH DECISION ===")
            print("Best action:", decision.best_action)

            print("\n=== ROOT TREE SNAPSHOT ===")
            for stat in decision.children_stats:
                print(
                    f"root -> {stat['action']['action']} "
                    f"| visits={stat['visits']} "
                    f"| mean_value={stat['mean_value']}"
                )
            
            if DEBUG_SEARCH:
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
            next_state = env.step(state, decision.best_action)
            
            trans_abs = extract_transition_abstraction(state.frame, next_state.frame)
            next_abs = extract_frame_abstraction(next_state.frame, include_cells=True)

            print("\n=== TRANSITION ABSTRACTION ===")
            print(summarize_for_llm(next_abs, trans_abs))

            if CNN_as_proposer: proposer.observe_action(state, decision.best_action, next_state)
            
            print("\n=== REAL TRANSITION ===")
            print("Action taken:", decision.best_action.action)
            print("Prev score:", state.score, "-> Next score:", next_state.score)
            print("Prev state:", state.state, "-> Next state:", next_state.state)
            print("Prev available:", state.available_actions)
            print("Next available:", next_state.available_actions)

            state = next_state
            step_counter += 1
        
        # Real committed actions in order
        print("\n=== EPISODE ACTION SEQUENCE ===")
        for i, action in enumerate(best_action_history):
            print(f"{i}: {action}")
        
        print("\n=== EPISODE ROOT CHOICES ===")
        for i, decision in enumerate(episode_decisions):
            print(f"Step {i}: best={decision.best_action.action}")
            for stat in decision.children_stats:
                print(
                    f"  root -> {stat['action']['action']} "
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