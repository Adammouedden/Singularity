import re
import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

# --- Config ---
model_id    = "google/t5gemma-2-270m-270m"
results_dir = "reasoning/singularis/benchmark/results/ARC-e"
batch_size  = 8
n_samples   = 10  # number of wrong + correct examples to keep for inspection
os.makedirs(results_dir, exist_ok=True)

# --- Load Model & Tokenizer ---
print(f"\n[DEBUG] Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[DEBUG] Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"[DEBUG] Model loaded on: {next(model.parameters()).device}")

# --- Dataset: AI2_ARC (Easy subset) ---
# Format: {'question': str, 'choices': {'text': [str], 'label': [str]}, 'answerKey': str}
dataset = load_dataset("ai2_arc", "ARC-Easy")["test"]
print(f"[DEBUG] Test set size: {len(dataset)}")

# --- Answer Extraction ---
def extract_arc_answer(text: str, valid_labels: list) -> str:
    text = text.strip().upper()
    match = re.search(r"ANSWER:\s*([A-D1-4])", text)
    if match:
        return match.group(1)
    for label in valid_labels:
        if text.startswith(label) or text.endswith(label):
            return label
    pattern = rf"[{''.join(valid_labels)}]"
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""

# --- Prompt Format ---
def format_prompt(item):
    question = item["question"]
    choices  = item["choices"]
    formatted_choices = "\n".join([f"({l}) {t}" for l, t in zip(choices["label"], choices["text"])])
    return (
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}\n"
        f"Answer the question by providing only the correct label (A, B, C, or D). Answer: "
    )

# --- Dry run before full loop ---
print("\n[DEBUG] --- DRY RUN on example 0 ---")
_ex     = dataset[0]
_prompt = format_prompt(_ex)
_inputs = tokenizer(_prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
with torch.no_grad():
    _out = model.generate(**_inputs, max_new_tokens=10, do_sample=False)
_pred = tokenizer.decode(_out[0], skip_special_tokens=True)
print(f"[DEBUG] Input:      {_prompt[:120]}...")
print(f"[DEBUG] Prediction: {_pred}")
print(f"[DEBUG] True label: {_ex['answerKey']}")
print(f"[DEBUG] Pred label: {extract_arc_answer(_pred, _ex['choices']['label'])}")
print("[DEBUG] --- DRY RUN complete ---\n")

# --- Collate ---
def collate_fn(batch):
    prompts     = [format_prompt(item) for item in batch]
    answer_keys = [item["answerKey"] for item in batch]
    labels_list = [item["choices"]["label"] for item in batch]
    return prompts, answer_keys, labels_list

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# --- Evaluation loop ---
correct        = 0
wrong_samples  = []   # up to n_samples incorrect examples
right_samples  = []   # up to n_samples correct examples
start_time     = datetime.now()
print(f"Starting evaluation at {start_time.strftime('%H:%M:%S')}...")

for i, (prompts, true_answers, all_labels) in enumerate(dataloader):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for j, prediction in enumerate(decoded_preds):
        idx        = i * batch_size + j
        pred_final = extract_arc_answer(prediction, all_labels[j])
        true_final = true_answers[j]
        is_correct = pred_final == true_final

        if is_correct:
            correct += 1

        sample = {
            "id":         idx,
            "true":       true_final,
            "pred":       pred_final,
            "raw_output": prediction[:200],
        }
        if not is_correct and len(wrong_samples) < n_samples:
            wrong_samples.append(sample)
        elif is_correct and len(right_samples) < n_samples:
            right_samples.append(sample)

    # Progress
    processed = min((i + 1) * batch_size, len(dataset))
    if processed % 80 == 0 or processed == len(dataset):
        elapsed = (datetime.now() - start_time).seconds
        acc     = correct / processed * 100
        print(f"[{processed:4d}/{len(dataset)}] Acc: {acc:.2f}% | Correct: {correct} | Elapsed: {elapsed}s")

# --- Save ---
end_time   = datetime.now()
duration_s = (end_time - start_time).total_seconds()
total      = len(dataset)
accuracy   = round(correct / total * 100, 2)

output = {
    "run": {
        "model_id":      model_id,
        "benchmark":     "ARC-Easy",
        "started_at":    start_time.isoformat(),
        "finished_at":   end_time.isoformat(),
        "duration_s":    round(duration_s, 1),
        "batch_size":    batch_size,
        "torch_dtype":   "bfloat16",
        "decoding":      "greedy",
        "prompt_format": "Question + choices → Answer: <label>",
        "notes":         "baseline — vanilla T5Gemma2-270m, no URM",
    },
    "summary": {
        "total":       total,
        "correct":     correct,
        "incorrect":   total - correct,
        "accuracy_pct": accuracy,
    },
    "samples": {
        "wrong": wrong_samples,
        "right": right_samples,
    },
}

timestamp = start_time.strftime("%Y%m%d_%H%M%S")
out_path  = os.path.join(results_dir, f"arc_easy_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(
    f"\nDONE."
    f"\n  Accuracy : {accuracy}%  ({correct}/{total})"
    f"\n  Duration : {duration_s:.1f}s"
    f"\n  Saved to : {out_path}"
)