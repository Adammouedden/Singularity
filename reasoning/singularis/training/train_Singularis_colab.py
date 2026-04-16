from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights
from reasoning.singularis.training.datasets import train_loader_combined, test_loader_combined, tokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput

import os
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import wandb

# Hyperparameters
LR      = 1e-3
EPOCHS  = 50
PATIENCE = 7
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

wandb.init(
    project="singularis",
    name="singularis-training-run-1",
    config={
        "lr": LR,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": 8,
        "max_length": 128,
        "optimizer": "AdamW",
        "datasets": ["gsm8k", "arc-easy", "arc-challenge"],
        "frozen": ["encoder", "decoder"],
        "trainable": ["urm_bridge"],
        "gradient_checkpointing": True,
    },
)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
model.load_weights(encoder_weights, decoder_weights)

# Freeze encoder and decoder — only the URM bridge trains
for param in model.singularis.encoder.parameters():
    param.requires_grad = False
for param in model.singularis.decoder.parameters():
    param.requires_grad = False
for param in model.singularis.urm_bridge.parameters():
    param.requires_grad = True

# Enable gradient checkpointing on the URM bridge if supported
if hasattr(model.singularis.urm_bridge, "gradient_checkpointing_enable"):
    model.singularis.urm_bridge.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled via HuggingFace API.")
elif hasattr(model.singularis.urm_bridge, "config"):
    model.singularis.urm_bridge.config.gradient_checkpointing = True
    print("Gradient checkpointing enabled via config.")
else:
    print("URM bridge does not natively support gradient checkpointing — will use torch.utils.checkpoint manually.")

# Optimizer only covers URM parameters
optimizer = torch.optim.AdamW(model.singularis.urm_bridge.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=-100)


def build_decoder_input(labels):
    """Shift labels right: [BOS, label_0, ..., label_N-2]. -100 pads become pad_token_id."""
    decoder_input_ids = labels.clone()
    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
    bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id, device=labels.device, dtype=torch.long)
    return torch.cat([bos, decoder_input_ids[:, :-1]], dim=1)


def urm_forward(urm_bridge, hidden_states):
    """
    Thin wrapper so torch.utils.checkpoint can call the URM bridge.
    checkpoint() requires a plain function with tensor inputs/outputs.
    """
    return urm_bridge(hidden_states)


def run_epoch(loader, train=True):
    model.singularis.encoder.eval()
    model.singularis.decoder.eval()
    model.singularis.urm_bridge.train() if train else model.singularis.urm_bridge.eval()

    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for step, batch in enumerate(loader):
            input_ids         = batch["input_ids"].to(device)
            attention_mask    = batch["attention_mask"].to(device)
            labels            = batch["labels"].to(device)
            decoder_input_ids = build_decoder_input(labels)

            # Run encoder under no_grad — frozen weights need no gradient tape
            with torch.no_grad():
                encoder_outputs = model.singularis.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            # Detach so the autograd graph starts fresh at the URM
            encoder_hidden_states = encoder_outputs.last_hidden_state.detach()

            # URM (trainable) — use gradient checkpointing during training to save VRAM
            if train:
                urm_hidden_states = checkpoint.checkpoint(
                    urm_forward,
                    model.singularis.urm_bridge,
                    encoder_hidden_states,
                    use_reentrant=False,  # recommended for modern PyTorch
                )
            else:
                urm_hidden_states = model.singularis.urm_bridge(encoder_hidden_states)

            with torch.no_grad():
                decoder_outputs = model.singularis.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=urm_hidden_states,
                    encoder_attention_mask=attention_mask,
                    return_dict=True,
                )

            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
            )

            logits = model.lm_head(outputs.last_hidden_state)   # [B, S, vocab_size]
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
        torch.save(
            model.singularis.urm_bridge.state_dict(),
            os.path.join(CHECKPOINT_DIR, "urm_best.pt"),
        )
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