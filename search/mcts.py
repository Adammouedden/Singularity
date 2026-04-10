"""
MCTS with:

expansion from LLM action proposals
rollout policy = random choice among valid actions or LLM proposals
value = simple heuristic from environment response
"""