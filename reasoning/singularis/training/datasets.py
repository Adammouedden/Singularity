from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from transformers import AutoTokenizer

MAX_LENGTH = 256
BATCH_SIZE = 16

# Load datasets
dataset_ai2_arc = load_dataset("allenai/ai2_arc", "ARC-Easy")
dataset_arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset_gsm8k = load_dataset("gsm8k", "main")

tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-448")

# PREPROCESS ARC GSM8K
def preprocess_gsm8k(example):
    # Input: the question
    # Target: full reasoning chain + answer
    inputs = tokenizer(
        "solve: " + example["question"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets = tokenizer(
        example["answer"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = targets["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {k: v.squeeze(0) for k, v in inputs.items()}

# PREPROCESS ARC DATASETS
def preprocess_arc(example):
    choices = example["choices"]
    formatted_choices = "\n".join(
        f"({l}) {t}" for l, t in zip(choices["label"], choices["text"])
    )
    prompt = (
        f"Question: {example['question']}\n"
        f"Choices:\n{formatted_choices}\n"
        f"Answer: "
    )
    inputs = tokenizer(
        prompt,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets = tokenizer(
        example["answerKey"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = targets["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {k: v.squeeze(0) for k, v in inputs.items()}

print("_______________________________________ BUILDING TRAIN/TEST DATA _______________________________________")

train_dataset_gsm8k = dataset_gsm8k["train"].map(preprocess_gsm8k, remove_columns=dataset_gsm8k["train"].column_names)
train_dataset_gsm8k.set_format("torch")

train_dataset_arc_easy = dataset_ai2_arc["train"].map(preprocess_arc, remove_columns=dataset_ai2_arc["train"].column_names)
train_dataset_arc_easy.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset_arc_challenge = dataset_arc_challenge["train"].map(preprocess_arc, remove_columns=dataset_arc_challenge["train"].column_names)
train_dataset_arc_challenge.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



# Datasets are of different sizes, and GSM8K accounts for 70% of the data, so we'll use weights to sample an uniform distribution of
# data from each dataset, ~33% each since we have replacement (reusing selected samples)
n_gsm8k = len(train_dataset_gsm8k)
n_arc_easy = len(train_dataset_arc_easy)
n_arc_chal = len(train_dataset_arc_challenge)
weights = (
    [1.0 / n_gsm8k] * n_gsm8k +
    [1.0 / n_arc_easy] * n_arc_easy +
    [1.0 / n_arc_chal] * n_arc_chal
)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_dataset_combined = ConcatDataset([train_dataset_gsm8k, train_dataset_arc_easy, train_dataset_arc_challenge])
train_loader_combined = DataLoader(train_dataset_combined, batch_size=BATCH_SIZE, sampler=sampler)

# Test splits — no weighted sampler, evaluate on true distribution
test_dataset_gsm8k = dataset_gsm8k["test"].map(preprocess_gsm8k, remove_columns=dataset_gsm8k["test"].column_names)
test_dataset_gsm8k.set_format("torch")

test_dataset_arc_easy = dataset_ai2_arc["test"].map(preprocess_arc, remove_columns=dataset_ai2_arc["test"].column_names)
test_dataset_arc_easy.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset_arc_challenge = dataset_arc_challenge["test"].map(preprocess_arc, remove_columns=dataset_arc_challenge["test"].column_names)
test_dataset_arc_challenge.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset_combined = ConcatDataset([test_dataset_gsm8k, test_dataset_arc_easy, test_dataset_arc_challenge])
test_loader_combined = DataLoader(test_dataset_combined, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    print(train_dataset_gsm8k)
    print(train_dataset_arc_easy)
    print(train_dataset_arc_challenge)
    print(train_loader_combined)
    print(test_dataset_gsm8k)
    print(test_dataset_arc_easy)
    print(test_dataset_arc_challenge)
    print(test_loader_combined)