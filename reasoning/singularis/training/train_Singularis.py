from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-448")

# initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
model.load_weights(encoder_weights, decoder_weights)

# Hyperparameters
LR = 1e-3
EPOCHS = 10
MAX_LENGTH = 256
BATCH_SIZE = 4
# Freeze the encoder weights 
for param in model.singularis.encoder.parameters():
    param.requires_grad = False

# Load the dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")

def preprocess(example):
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
    inputs["labels"] = targets["input_ids"]
    return {k: v.squeeze(0) for k, v in inputs.items()}
    
train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
train_dataset.set_format("torch")
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

