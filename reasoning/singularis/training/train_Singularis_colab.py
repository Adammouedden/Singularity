"""
Singularis — Colab training script
===================================
Trains URM bridge + LoRA adapters on the decoder using ARC (Easy + Challenge).

Colab-resilience features
--------------------------
  * Automatic resume from the latest checkpoint on startup.
  * Mid-epoch checkpoint every SAVE_EVERY_STEPS steps — survives session timeouts
    mid-epoch.
  * End-of-epoch checkpoint so a full epoch is never lost.
  * All checkpoint state includes: epoch, global_step, URM weights, LoRA weights,
    optimizer state, best_val_loss, patience_counter.

Google Drive (recommended for Colab)
--------------------------------------
  Mount Drive and point CHECKPOINT_DIR at it so checkpoints survive runtime resets:

      from google.colab import drive
      drive.mount('/content/drive')

  Then set:  CHECKPOINT_DIR = "/content/drive/MyDrive/singularis_checkpoints"
"""

import glob
import math
import os
import time

import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_checkpoint
import wandb
from torch.utils.data import ConcatDataset, DataLoader

from reasoning.singularis.config_and_weights import (
    LLM_config, URM_config, decoder_weights, encoder_weights,
)
from reasoning.singularis.model import SingularisForConditionalGeneration


# ─── LoRA ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank adapters A and B.

    Output = W·x + (B·A·x) * (alpha/rank)

    B is zero-initialized so the adapter is a no-op at the start of training,
    preserving the pretrained baseline.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank

        in_f, out_f = linear.in_features, linear.out_features
        dtype, device = linear.weight.dtype, linear.weight.device

        self.lora_A = nn.Linear(in_f,  rank,  bias=False).to(device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank,  out_f, bias=False).to(device=device, dtype=dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(
    module: nn.Module,
    target_suffixes: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Replace every nn.Linear whose attr name ends with a target suffix with
    LoRALinear.  Call AFTER freezing base weights.  Returns adapter count."""
    replaced = 0
    for parent in module.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(
                child_name.endswith(s) for s in target_suffixes
            ):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
    return replaced


def lora_parameters(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def count_parameters(module: nn.Module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ─── Hyperparameters ─────────────────────────────────────────────────────────

LORA_RANK    = 8
LORA_ALPHA   = 16.0
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

LR               = 1e-3
EPOCHS           = 50
PATIENCE         = 7
BATCH_SIZE       = 8
SAVE_EVERY_STEPS = 200   # mid-epoch saves for Colab timeout resilience

CHECKPOINT_DIR = "checkpoints/lora_arc"
# CHECKPOINT_DIR = "/content/drive/MyDrive/singularis_checkpoints"  # ← for Colab + Drive


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_checkpoint(path, *, epoch, global_step, model, optimizer,
                    best_val_loss, patience_counter):
    lora_state = {
        name: param.data.clone()
        for name, param in model.singularis.decoder.named_parameters()
        if param.requires_grad
    }
    torch.save({
        "epoch":            epoch,
        "global_step":      global_step,
        "urm_bridge":       model.singularis.urm_bridge.state_dict(),
        "decoder_lora":     lora_state,
        "optimizer":        optimizer.state_dict(),
        "best_val_loss":    best_val_loss,
        "patience_counter": patience_counter,
    }, path)


def find_latest_checkpoint(checkpoint_dir: str):
    """Return the path of the latest checkpoint, preferring epoch > step saves."""
    epoch_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")))
    step_ckpts  = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt")))
    if epoch_ckpts and step_ckpts:
        # Return whichever was modified most recently
        latest_epoch = epoch_ckpts[-1]
        latest_step  = step_ckpts[-1]
        return latest_epoch if os.path.getmtime(latest_epoch) >= os.path.getmtime(latest_step) else latest_step
    return (epoch_ckpts or step_ckpts or [None])[-1]


def load_checkpoint(path, *, model, optimizer, device):
    """Restore model + optimizer.  Returns (epoch, global_step, best_val_loss, patience_counter)."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.singularis.urm_bridge.load_state_dict(state["urm_bridge"])

    dec_params = dict(model.singularis.decoder.named_parameters())
    for name, data in state["decoder_lora"].items():
        if name in dec_params:
            dec_params[name].data.copy_(data)

    optimizer.load_state_dict(state["optimizer"])
    return (
        state["epoch"],
        state["global_step"],
        state["best_val_loss"],
        state["patience_counter"],
    )


# ─── Training helpers ────────────────────────────────────────────────────────

def build_decoder_input(labels, tokenizer):
    dec = labels.clone()
    dec[dec == -100] = tokenizer.pad_token_id
    bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id,
                     device=labels.device, dtype=torch.long)
    return torch.cat([bos, dec[:, :-1]], dim=1)


def _urm_forward(urm_bridge, hidden_states):
    """Plain-function wrapper required by torch.utils.checkpoint."""
    return urm_bridge(hidden_states)


def run_epoch(
    model, loader, optimizer, criterion, tokenizer, device,
    *,
    train: bool,
    global_step: int = 0,
    skip_steps: int = 0,
    checkpoint_dir: str | None = None,
    best_val_loss: float = float("inf"),
    patience_counter: int = 0,
    epoch: int = 0,
):
    model.singularis.encoder.eval()
    model.singularis.decoder.train() if train else model.singularis.decoder.eval()
    model.singularis.urm_bridge.train() if train else model.singularis.urm_bridge.eval()

    total_loss     = 0.0
    correct_tokens = 0
    total_tokens   = 0
    counted_steps  = 0
    context        = torch.enable_grad() if train else torch.no_grad()

    with context:
        for step, batch in enumerate(loader):
            # Resume mid-epoch: skip already-processed steps
            if step < skip_steps:
                continue

            input_ids         = batch["input_ids"].to(device)
            attention_mask    = batch["attention_mask"].to(device)
            labels            = batch["labels"].to(device)
            decoder_input_ids = build_decoder_input(labels, tokenizer)

            # Encoder — frozen, no gradient tape
            with torch.no_grad():
                encoder_outputs = model.singularis.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            encoder_hidden_states = encoder_outputs.last_hidden_state.detach()

            # URM bridge — gradient checkpointing saves activation memory
            if train:
                urm_hidden_states = grad_checkpoint.checkpoint(
                    _urm_forward,
                    model.singularis.urm_bridge,
                    encoder_hidden_states,
                    use_reentrant=False,
                )
            else:
                urm_hidden_states = model.singularis.urm_bridge(encoder_hidden_states)

            # Decoder — base weights frozen, LoRA adapters are trainable.
            # No no_grad here so gradients flow through LoRA layers.
            # use_cache=False is required when gradient_checkpointing is active.
            decoder_outputs = model.singularis.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=urm_hidden_states,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            logits = model.lm_head(decoder_outputs.last_hidden_state)
            loss   = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            mask = labels != -100
            preds = logits.argmax(dim=-1)
            correct_tokens += (preds[mask] == labels[mask]).sum().item()
            total_tokens   += mask.sum().item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % 100 == 0:
                    step_acc = (preds[mask] == labels[mask]).float().mean().item()
                    print(f"  Step {step:>5} | Loss: {loss.item():.4f} | Acc: {step_acc:.3f}")
                wandb.log({"train/step_loss": loss.item(), "global_step": global_step})

                # Mid-epoch checkpoint for Colab timeout resilience
                if checkpoint_dir and global_step % SAVE_EVERY_STEPS == 0:
                    mid_path = os.path.join(checkpoint_dir, f"ckpt_step_{global_step:07d}.pt")
                    save_checkpoint(
                        mid_path, epoch=epoch, global_step=global_step,
                        model=model, optimizer=optimizer,
                        best_val_loss=best_val_loss, patience_counter=patience_counter,
                    )
                    print(f"  [mid-epoch] Saved checkpoint → {mid_path}")

            total_loss    += loss.item()
            counted_steps += 1

    avg_loss = total_loss / max(counted_steps, 1)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy, global_step


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    from reasoning.singularis.training.datasets import (
        test_dataset_arc_challenge,
        test_dataset_arc_easy,
        tokenizer,
        train_loader_ARC_only,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ARC-only validation loader (mirrors training split)
    test_dataset_ARC_only = ConcatDataset([test_dataset_arc_easy, test_dataset_arc_challenge])
    test_loader_ARC_only  = DataLoader(test_dataset_ARC_only, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
    model.load_weights(encoder_weights, decoder_weights)

    # Freeze encoder completely
    for p in model.singularis.encoder.parameters():
        p.requires_grad = False

    # Freeze all base decoder weights BEFORE injecting LoRA so the pretrained
    # Linear inside each LoRALinear stays frozen
    for p in model.singularis.decoder.parameters():
        p.requires_grad = False

    # Inject LoRA — new lora_A / lora_B params default to requires_grad=True
    n_adapters = inject_lora(model.singularis.decoder, LORA_TARGETS, LORA_RANK, LORA_ALPHA)

    # URM bridge fully trainable
    for p in model.singularis.urm_bridge.parameters():
        p.requires_grad = True

    # Gradient checkpointing on the decoder — saves activation memory across
    # all 18 layers; incompatible with use_cache=True (we pass use_cache=False)
    if hasattr(model.singularis.decoder, "gradient_checkpointing_enable"):
        model.singularis.decoder.gradient_checkpointing_enable()
        print("Decoder gradient checkpointing: enabled")
    else:
        print("Decoder gradient checkpointing: not supported by this module")

    total, trainable = count_parameters(model)
    print(f"Injected {n_adapters} LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    trainable_params = (
        list(model.singularis.urm_bridge.parameters())
        + list(lora_parameters(model.singularis.decoder))
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch      = 0
    global_step      = 0
    skip_steps       = 0
    best_val_loss    = float("inf")
    patience_counter = 0

    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"\nResuming from checkpoint: {latest_ckpt}")
        start_epoch, global_step, best_val_loss, patience_counter = load_checkpoint(
            latest_ckpt, model=model, optimizer=optimizer, device=device,
        )
        if "ckpt_step_" in os.path.basename(latest_ckpt):
            # Mid-epoch save: resume at the step within the current epoch
            steps_per_epoch = len(train_loader_ARC_only)
            skip_steps = global_step % steps_per_epoch
            print(f"  Mid-epoch resume — epoch {start_epoch + 1}, skipping first {skip_steps} steps")
        else:
            # End-of-epoch save: start the next epoch fresh
            start_epoch += 1
            print(f"  Epoch resume — starting epoch {start_epoch + 1}")
        print(f"  global_step={global_step}  best_val_loss={best_val_loss:.4f}  patience={patience_counter}")
    else:
        print("No checkpoint found — starting fresh.")

    # ── wandb (resume="allow" continues the same run after a Colab restart) ─
    wandb.init(
        project="singularis",
        name="singularis-lora-arc",
        resume="allow",
        config={
            "lr": LR, "epochs": EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "save_every_steps": SAVE_EVERY_STEPS,
            "datasets": ["arc-easy", "arc-challenge"],
            "frozen": ["encoder", "decoder_base"],
            "trainable": ["urm_bridge", "decoder_lora"],
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "lora_targets": list(LORA_TARGETS),
            "gradient_checkpointing": ["urm_bridge", "decoder"],
        },
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print("_______________________________________ TRAINING LOOP BEGINS _______________________________________")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        t0 = time.time()

        train_loss, train_acc, global_step = run_epoch(
            model, train_loader_ARC_only, optimizer, criterion, tokenizer, device,
            train=True,
            global_step=global_step,
            skip_steps=skip_steps,
            checkpoint_dir=CHECKPOINT_DIR,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            epoch=epoch,
        )
        skip_steps = 0  # only relevant for the first (resumed) epoch

        val_loss, val_acc, _ = run_epoch(
            model, test_loader_ARC_only, optimizer, criterion, tokenizer, device,
            train=False,
        )

        elapsed = time.time() - t0
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Time: {elapsed:.1f}s"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss, "train/epoch_acc": train_acc,
            "val/epoch_loss":   val_loss,   "val/epoch_acc":   val_acc,
            "epoch_time_s":     elapsed,
        })

        # End-of-epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch + 1:03d}.pt")
        save_checkpoint(
            ckpt_path, epoch=epoch, global_step=global_step,
            model=model, optimizer=optimizer,
            best_val_loss=best_val_loss, patience_counter=patience_counter,
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
            save_checkpoint(
                best_path, epoch=epoch, global_step=global_step,
                model=model, optimizer=optimizer,
                best_val_loss=best_val_loss, patience_counter=patience_counter,
            )
            print(f"  Saved best model → {best_path}  (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_val_acc"]  = val_acc
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
