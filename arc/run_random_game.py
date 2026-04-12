#!/usr/bin/env python3
"""
This is the ARC enviroment runner.
"""

import os
import random
import requests
from dotenv import load_dotenv
from world_models.gemini_world_model import GeminiWorldModel
# Load environment variables
load_dotenv(dotenv_path=".env")

# Setup
ROOT_URL = "https://arcprize.org"
API_KEY = os.getenv("ARC_API_KEY")
NUM_ACTIONS = 2 # 5
# Create a session with headers
session = requests.Session()
session.headers.update({
    "X-API-Key": API_KEY,
    "Accept": "application/json"
})

# Step 1: Get available games
print("STEP 1: Getting list of games...")
response = session.get(f"{ROOT_URL}/api/games")
games = [g["game_id"] for g in response.json()]
print(f"Found {len(games)} games")

# Pick a random game
game_id = "ls20-9607627b" #random.choice(games)
print(f"Selected game: {game_id}\n")

# Step 2: Open a scorecard (tracks performance)
print("STEP 2: Opening scorecard...")
response = session.post(
    f"{ROOT_URL}/api/scorecard/open",
    json={"tags": ["manual_demo"]}
)
card_id = response.json()["card_id"]
print(f"Scorecard ID: {card_id}\n")

# Step 3: Start the game
print("STEP 3: Starting game with RESET action...")
url = f"{ROOT_URL}/api/cmd/RESET"
print(f"URL: {url}")
response = session.post(
    url,
    json={
        "game_id": game_id,
        "card_id": card_id
    }
)

# Check if response is valid
if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
    exit()


game_data = response.json()
actions = game_data["available_actions"] # List of ints
guid = game_data["guid"]
state = game_data["state"]
score = game_data.get("score", 0)
print(f"Game started! State: {state}, Score: {score}\n")

# Step 4: Play with Gemini-proposed actions
print("STEP 4: Taking Gemini-proposed actions...")

world_model = GeminiWorldModel()

for i in range(NUM_ACTIONS):
    if state in ["WIN", "GAME_OVER"]:
        print(f"\nGame ended! Final state: {state}, Score: {score}")
        break

    frame = game_data["frame"]

    try:
        proposals = world_model.propose_actions(frame, actions)
    except Exception as e:
        print(f"Gemini proposal failed: {e}")
        break

    print(f"\nTurn {i+1} candidate actions:")
    for idx, candidate in enumerate(proposals.candidates, start=1):
        print(
            f"  {idx}. action={candidate.action}, x={candidate.x}, y={candidate.y}, rationale={candidate.rationale}"
        )

    # For now, just pick the first candidate
    chosen = proposals.candidates[0]

    if chosen.action not in [f"ACTION{a}" for a in actions]:
        print("Invalid action from LLM, falling back to random")
        chosen.action = f"ACTION{random.choice(actions)}"

    request_data = {
        "game_id": game_id,
        "card_id": card_id,
        "guid": guid,
        #"rationale":candidate.rationale # TODO, does not work yet.
    }

    action = chosen.action

    if action == "ACTION6":
        request_data["x"] = chosen.x if chosen.x is not None else random.randint(0, 63)
        request_data["y"] = chosen.y if chosen.y is not None else random.randint(0, 63)
        print(f"Chosen action: {action} at ({request_data['x']}, {request_data['y']})")
    else:
        print(f"Chosen action: {action}")

    response = session.post(
        f"{ROOT_URL}/api/cmd/{action}",
        json=request_data
    )

    if response.status_code != 200:
        print(f"Action failed: {response.status_code} - {response.text}")
        break

    game_data = response.json()
    state = game_data["state"]
    score = game_data.get("score", 0)
    guid = game_data["guid"]

    print(f" -> State: {state}, Score: {score}")

# Step 5: Close scorecard
print("\nSTEP 5: Closing scorecard...")
response = session.post(
    f"{ROOT_URL}/api/scorecard/close",
    json={"card_id": card_id}
)
scorecard = response.json()
print("Scorecard closed!")
print(f"\nView results at: {ROOT_URL}/scorecards/{card_id}")
