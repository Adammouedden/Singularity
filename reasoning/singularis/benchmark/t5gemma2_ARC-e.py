import re
import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

# --- Config ---
model_id    = "google/gemma-2-2b-it" # Updated to a causal model for easier formatting
results_dir = "./results"
batch_size  = 16  # ARC-e is lighter than GSM8K; you can push this on your setup
os.makedirs(results_dir, exist_ok=True)

# --- Load Model & Tokenizer ---
print(f"\n[DEBUG] Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[DEBUG] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# --- Dataset: AI2_ARC (Easy subset) ---
# Format: {'question': str, 'choices': {'text': [str], 'label': [str]}, 'answerKey': str}
dataset = load_dataset("ai2_arc", "ARC-Easy")["test"]

# --- ARC-Specific Answer Extraction ---
def extract_arc_answer(text: str, valid_labels: list) -> str:
    """Finds the choice label in the model's response."""
    # Look for patterns like "The answer is (A)" or just "A"
    text = text.strip().upper()
    
    # Try searching for "Answer: X"
    match = re.search(r"ANSWER:\s*([A-D1-4])", text)
    if match:
        return match.group(1)
    
    # Fallback: check if any valid label appears alone at the start or end
    for label in valid_labels:
        if text.startswith(label) or text.endswith(label):
            return label
            
    # Last resort: just find the first valid label mentioned
    pattern = rf"[{''.join(valid_labels)}]"
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""

# --- Prompt Formatting for Multiple Choice ---
def format_prompt(item):
    question = item["question"]
    choices = item["choices"]
    formatted_choices = "\n".join([f"({l}) {t}" for l, t in zip(choices["label"], choices["text"])])
    
    return (
        f"<start_of_turn>user\n"
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}\n"
        f"Identify the correct label (e.g., A, B, C, or D). Provide only the label after 'Answer: '.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"Answer: "
    )

# --- Evaluation Loop ---
results = []
correct = 0
start_time = datetime.now()

def collate_fn(batch):
    prompts = [format_prompt(item) for item in batch]
    answer_keys = [item["answerKey"] for item in batch]
    labels_list = [item["choices"]["label"] for item in batch]
    return prompts, answer_keys, labels_list

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

print(f"Starting ARC-e evaluation at {start_time.strftime('%H:%M:%S')}...")

for i, (prompts, true_answers, all_labels) in enumerate(dataloader):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10, # Very short for ARC MCQs
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the NEW tokens (skip the prompt)
    input_len = inputs.input_ids.shape[1]
    decoded_preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    for j, prediction in enumerate(decoded_preds):
        idx = i * batch_size + j
        pred_label = extract_arc_answer(prediction, all_labels[j])
        true_label = true_answers[j]

        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1

        results.append({
            "id": idx,
            "true_answer": true_label,
            "pred_label": pred_label,
            "correct": is_correct,
            "raw_output": prediction
        })

    if (idx + 1) % batch_size == 0:
        acc = (correct / (idx + 1)) * 100
        print(f"[{idx+1}/{len(dataset)}] Acc: {acc:.2f}%")

# --- Final Metrics ---
duration = (datetime.now() - start_time).seconds
metrics = {
    "model_id": model_id,
    "accuracy": round(correct / len(dataset) * 100, 2),
    "total": len(dataset),
    "correct": correct,
    "duration_s": duration
}

timestamp = start_time.strftime("%Y%m%d_%H%M%S")
with open(os.path.join(results_dir, f"arc_metrics_{timestamp}.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nDONE. Final Accuracy: {metrics['accuracy']}%")