import os
import json

results_dir = "reasoning/singularis/benchmark/results/ARC-e"
batch_size  = 16  # ARC-e is lighter than GSM8K; you can push this on your setup
os.makedirs(results_dir, exist_ok=True)

out_path = os.path.join(results_dir, f"arc_easy_101101.json")
with open(out_path, "w") as f:
    json.dump({}, f, indent=2)
