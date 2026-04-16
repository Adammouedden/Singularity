import os
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import wandb
from peft import LoraConfig, get_peft_model, TaskType

from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights
from reasoning.singularis.training.datasets import train_loader_combined, test_loader_combined, tokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput

# Hyperparameters
LR       = 1e-3
EPOCHS   = 50
PATIENCE = 7
CHECKPOINT_DIR = "checkpoints"

# LoRA Configuration
# Note: target_modules may need to be adjusted based on your specific decoder's architecture
# (e.g., ["q_proj", "v_proj"] for Llama-style, ["query", "value"] for others)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM 
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

wandb.init(
    project="singularis",
    name="singularis-lora-training",
    config={
        "lr": LR,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "trainable": ["urm_bridge", "decoder_lora", "lm_head"],
        "gradient_checkpointing": True,
    },
)

# --- Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
model.load_weights(encoder_weights, decoder_weights)

# 1. Apply LoRA to the decoder
# This freezes the base weights of the decoder and adds trainable adapter layers
model.singularis.decoder = get_peft_model(model.singularis.decoder, lora_config)

# 2. Parameter Freezing Logic
# Freeze encoder
for param in model.singularis.encoder.parameters():
    param.requires_grad = False

# Ensure URM bridge is trainable
for param in model.singularis.urm_bridge.parameters():
    param.requires_grad = True

# Ensure LM Head is trainable (often beneficial when fine-tuning the decoder)
for param in model.lm_head.parameters():
    param.requires_grad = True

# 3. Gradient Checkpointing
if hasattr(model.singularis.urm_bridge, "gradient_checkpointing_enable"):
    model.singularis.urm_bridge.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled on URM bridge.")

# 4. Optimizer
# Include URM bridge, LoRA adapters, and LM head
trainable_params = [
    {"params": model.singularis.urm_bridge.parameters()},
    {"params": model.singularis.decoder.parameters()},
    {"params": model.lm_head.parameters()}
]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=-100)


def build_decoder_input(labels):
    """Shift labels right: [BOS, label_0, ..., label_N-2]. -100 pads become pad_token_id."""
    decoder_input_ids = labels.clone()
    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
    bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id, device=labels.device, dtype=torch.long)
    return torch.cat([bos, decoder_input_ids[:, :-1]], dim=1)


def urm_forward(urm_bridge, hidden_states):
    return urm_bridge(hidden_states)


def run_epoch(loader, train=True):
    # Encoder always in eval (frozen)
    model.singularis.encoder.eval()
    
    # URM and Decoder (LoRA) toggle based on phase
    if train:
        model.singularis.urm_bridge.train()
        model.singularis.decoder.train()
        model.lm_head.train()
    else:
        model.singularis.urm_bridge.eval()
        model.singularis.decoder.eval()
        model.lm_head.eval()

    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for step, batch in enumerate(loader):
            input_ids         = batch["input_ids"].to(device)
            attention_mask    = batch["attention_mask"].to(device)
            labels            = batch["labels"].to(device)
            decoder_input_ids = build_decoder_input(labels)

            # Encoder (Frozen) - always no_grad
            with torch.no_grad():
                encoder_outputs = model.singularis.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            encoder_hidden_states = encoder_outputs.last_hidden_state.detach()

            # URM Bridge (Trainable)
            if train:
                urm_hidden_states = checkpoint.checkpoint(
                    urm_forward,
                    model.singularis.urm_bridge,
                    encoder_hidden_states,
                    use_reentrant=False,
                )
            else:
                urm_hidden_states = model.singularis.urm_bridge(encoder_hidden_states)

            # Decoder (LoRA - Trainable)
            # Gradient context managed by the 'with context' block above
            decoder_outputs = model.singularis.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=urm_hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )

            # LM Head
            logits = model.lm_head(decoder_outputs.last_hidden_state)
            loss   = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step % 100 == 0:
                    print(f"  Step {step:>4} | Loss: {loss.item():.4f}")
                wandb.log({"train/step_loss": loss.item()})

            total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss    = float("inf")
patience_counter = 0

print("_______________________________________ TRAINING LOOP BEGINS _______________________________________")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    t0 = time.time()

    train_loss = run_epoch(train_loader_combined, train=True)
    val_loss   = run_epoch(test_loader_combined,  train=False)

    elapsed = time.time() - t0
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    wandb.log({
        "epoch": epoch + 1,
        "train/epoch_loss": train_loss,
        "val/epoch_loss": val_loss,
        "epoch_time_s": elapsed,
    })

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        
        # Save URM weights and LoRA adapters separately for cleanliness
        torch.save(model.singularis.urm_bridge.state_dict(), os.path.join(CHECKPOINT_DIR, "urm_best.pt"))
        model.singularis.decoder.save_pretrained(os.path.join(CHECKPOINT_DIR, "decoder_lora_best"))
        
        print(f"  Saved best model (val_loss={val_loss:.4f})")
        wandb.summary["best_val_loss"] = best_val_loss
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
wandb.finish()