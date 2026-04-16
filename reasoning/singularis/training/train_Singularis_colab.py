import os
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.amp import autocast, GradScaler # Added AMP
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from transformers import get_linear_schedule_with_warmup # Added Scheduler

from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights
from reasoning.singularis.training.datasets import train_loader_combined, test_loader_combined, tokenizer

# --- Hyperparameters ---
LR = 2e-4 # Lowered for LoRA stability
EPOCHS = 50
PATIENCE = 7
GRAD_ACCUM_STEPS = 4  # Effective batch size = 8 * 4 = 32
WARMUP_STEPS = 100
CHECKPOINT_DIR = "checkpoints"

# --- LoRA Config ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # Ensure these names match the actual attributes in your T5Gemma2Decoder
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type=None  # <--- Change this from TaskType.CAUSAL_LM
)

wandb.init(
    project="singularis",
    name="singularis-refined-lora-GSM8K-and-ARC",
    config={
        "lr": LR,
        "effective_batch_size": 8 * GRAD_ACCUM_STEPS,
        "lora_r": 32,
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
model.load_weights(encoder_weights, decoder_weights)

# Model Setup
model.singularis.decoder = get_peft_model(model.singularis.decoder, lora_config)
for param in model.singularis.encoder.parameters(): param.requires_grad = False
for param in model.singularis.urm_bridge.parameters(): param.requires_grad = True
for param in model.lm_head.parameters(): param.requires_grad = True

# Optimizer & Scheduler
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

total_steps = len(train_loader_combined) * EPOCHS // GRAD_ACCUM_STEPS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
scaler = GradScaler() # For AMP

criterion = nn.CrossEntropyLoss(ignore_index=-100)

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0

def run_epoch(loader, train=True):
    model.singularis.encoder.eval() # Always frozen
    if train:
        model.singularis.urm_bridge.train()
        model.singularis.decoder.train()
        model.lm_head.train()
    else:
        model.singularis.urm_bridge.eval()
        model.singularis.decoder.eval()
        model.lm_head.eval()
    
    total_loss, total_acc = 0.0, 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Shift labels for causal input
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
        
        # Use bfloat16 for stability on modern GPUs (no GradScaler needed)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                encoder_outs = model.singularis.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            
            urm_outs = model.singularis.urm_bridge(encoder_outs)
            
            decoder_results = model.singularis.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=urm_outs,
                encoder_attention_mask=attention_mask
            )
            
            # Extract hidden states safely
            h = decoder_results.last_hidden_state if hasattr(decoder_results, 'last_hidden_state') else decoder_results
            
            logits = model.lm_head(h)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / GRAD_ACCUM_STEPS

        if train:
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Metrics calculation
        acc = calculate_accuracy(logits, labels)
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        total_acc += acc

        if step % 100 == 0:
            print(f"{'Train' if train else 'Val'} Step {step:>4} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | Acc: {acc:.4f}")

    return total_loss / len(loader), total_acc / len(loader)

# --- Loop ---
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    t_loss, t_acc = run_epoch(train_loader_combined, train=True)
    v_loss, v_acc = run_epoch(test_loader_combined, train=False)

    print(f"Epoch {epoch+1} | Val Acc: {v_acc:.4f} | Val Loss: {v_loss:.4f}")
    
    wandb.log({"val_acc": v_acc, "val_loss": v_loss, "train_loss": t_loss})

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE: break