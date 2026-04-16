import math
import os
import time

import torch
import torch.nn as nn

from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights


# ─── LoRA ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank adapters A and B.

    Output = W·x + (B·A·x) * (alpha/rank)

    B is zero-initialized so the adapter is a no-op at the start of training,
    preserving the pretrained baseline.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear  = linear         # frozen pretrained weights
        self.scaling = alpha / rank

        in_f, out_f = linear.in_features, linear.out_features
        dtype  = linear.weight.dtype
        device = linear.weight.device

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
    """Walk the module tree and replace every nn.Linear whose attribute name
    ends with a target suffix with a LoRALinear.  Call this AFTER freezing
    base weights so the pretrained Linear inside each LoRALinear stays frozen
    while the new lora_A / lora_B params default to requires_grad=True.

    Returns the number of adapters injected.
    """
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
    """Yield only the LoRA adapter parameters (lora_A / lora_B) inside module."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def count_parameters(module: nn.Module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ─── LoRA hyper-parameters ───────────────────────────────────────────────────

LORA_RANK    = 8
LORA_ALPHA   = 16.0
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# ─── Training helpers ────────────────────────────────────────────────────────

def build_decoder_input(labels, tokenizer):
    """Shift labels right: [BOS, label_0, ..., label_N-2]. -100 pads → pad_token_id."""
    decoder_input_ids = labels.clone()
    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
    bos = torch.full(
        (labels.size(0), 1), tokenizer.bos_token_id,
        device=labels.device, dtype=torch.long,
    )
    return torch.cat([bos, decoder_input_ids[:, :-1]], dim=1)


def run_epoch(model, loader, optimizer, criterion, tokenizer, device, train=True):
    # Always keep the encoder weights frozen
    model.singularis.encoder.eval()
    # Decoder base weights are frozen; LoRA adapters toggle with train/eval
    model.singularis.decoder.train() if train else model.singularis.decoder.eval()
    #Train the URM Bridge
    model.singularis.urm_bridge.train() if train else model.singularis.urm_bridge.eval()

    total_loss     = 0.0
    correct_tokens = 0
    total_tokens   = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for step, batch in enumerate(loader):
            input_ids         = batch["input_ids"].to(device)
            attention_mask    = batch["attention_mask"].to(device)
            labels            = batch["labels"].to(device)
            decoder_input_ids = build_decoder_input(labels, tokenizer)

            # Encoder — frozen, no gradient tape needed
            with torch.no_grad():
                encoder_outputs = model.singularis.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            encoder_hidden_states = encoder_outputs.last_hidden_state.detach()

            # URM bridge (trainable)
            urm_hidden_states = model.singularis.urm_bridge(encoder_hidden_states)

            # Decoder — base weights frozen, LoRA adapters are trainable
            # No torch.no_grad() here so gradients flow through the LoRA layers
            decoder_outputs = model.singularis.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=urm_hidden_states,
                encoder_attention_mask=attention_mask,
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
                if step % 100 == 0:
                    step_acc = (preds[mask] == labels[mask]).float().mean().item()
                    print(f"  Step {step:>4} | Loss: {loss.item():.4f} | Acc: {step_acc:.3f}")

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train():
    """Full training run: URM bridge + LoRA on decoder attention."""
    import wandb
    from reasoning.singularis.training.datasets import (
        train_loader_combined, test_loader_combined, tokenizer,
    )

    LR             = 1e-3
    EPOCHS         = 50
    PATIENCE       = 7
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project="singularis",
        name="singularis-lora-run-1",
        config={
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": 8,
            "max_length": 128,
            "optimizer": "AdamW",
            "datasets": ["gsm8k", "arc-easy", "arc-challenge"],
            "frozen": ["encoder", "decoder_base"],
            "trainable": ["urm_bridge", "decoder_lora"],
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_targets": list(LORA_TARGETS),
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
    model.load_weights(encoder_weights, decoder_weights)

    # Freeze encoder completely
    for p in model.singularis.encoder.parameters():
        p.requires_grad = False

    # Freeze all base decoder weights BEFORE injecting LoRA so the pretrained
    # Linear wrapped inside each LoRALinear stays frozen
    for p in model.singularis.decoder.parameters():
        p.requires_grad = False

    # Inject LoRA — creates new trainable lora_A / lora_B for each attention proj
    n_adapters = inject_lora(model.singularis.decoder, LORA_TARGETS, LORA_RANK, LORA_ALPHA)

    # URM bridge stays fully trainable
    for p in model.singularis.urm_bridge.parameters():
        p.requires_grad = True

    total, trainable = count_parameters(model)
    print(f"Injected {n_adapters} LoRA adapters into decoder")
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    trainable_params = (
        list(model.singularis.urm_bridge.parameters())
        + list(lora_parameters(model.singularis.decoder))
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss    = float("inf")
    patience_counter = 0

    print("_______________________________________ TRAINING LOOP BEGINS _______________________________________")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader_combined, optimizer, criterion, tokenizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, test_loader_combined,  optimizer, criterion, tokenizer, device, train=False)

        elapsed = time.time() - t0
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Time: {elapsed:.1f}s"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "train/epoch_acc":  train_acc,
            "val/epoch_loss":   val_loss,
            "val/epoch_acc":    val_acc,
            "epoch_time_s":     elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0

            # Save URM bridge + LoRA adapter weights together
            lora_state = {
                name: param
                for name, param in model.singularis.decoder.named_parameters()
                if param.requires_grad
            }
            torch.save(
                {
                    "urm_bridge":   model.singularis.urm_bridge.state_dict(),
                    "decoder_lora": lora_state,
                },
                os.path.join(CHECKPOINT_DIR, "best.pt"),
            )
            print(f"  Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
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


# ─── Single-sample LoRA smoke test ───────────────────────────────────────────

if __name__ == "__main__":
    print("=== LoRA single-sample smoke test ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model with pretrained weights
    model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device)
    model.load_weights(encoder_weights, decoder_weights)

    # Freeze encoder and base decoder weights
    for p in model.singularis.encoder.parameters():
        p.requires_grad = False
    for p in model.singularis.decoder.parameters():
        p.requires_grad = False
    for p in model.singularis.urm_bridge.parameters():
        p.requires_grad = True

    # Inject LoRA into decoder self-attention projections
    n_adapters = inject_lora(model.singularis.decoder, LORA_TARGETS, LORA_RANK, LORA_ALPHA)
    print(f"Injected {n_adapters} LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")

    total, trainable = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)\n")

    # Synthetic single-sample batch (no real data needed)
    BATCH   = 1
    ENC_SEQ = 32
    DEC_SEQ = 16
    VOCAB   = 255_999   # below T5Gemma2 special tokens

    input_ids         = torch.randint(0, VOCAB, (BATCH, ENC_SEQ), dtype=torch.long,  device=device)
    attention_mask    = torch.ones(BATCH, ENC_SEQ,              dtype=torch.long,  device=device)
    decoder_input_ids = torch.randint(0, VOCAB, (BATCH, DEC_SEQ), dtype=torch.long,  device=device)
    labels            = torch.randint(0, VOCAB, (BATCH, DEC_SEQ), dtype=torch.long,  device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.singularis.urm_bridge.parameters())
        + list(lora_parameters(model.singularis.decoder)),
        lr=1e-3,
    )

    # ── Forward ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        encoder_outputs = model.singularis.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    encoder_hidden_states = encoder_outputs.last_hidden_state.detach()

    urm_hidden_states = model.singularis.urm_bridge(encoder_hidden_states)

    # Decoder — no torch.no_grad() so LoRA adapters receive gradients
    decoder_outputs = model.singularis.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=urm_hidden_states,
        encoder_attention_mask=attention_mask,
        return_dict=True,
    )

    logits = model.lm_head(decoder_outputs.last_hidden_state)   # [1, DEC_SEQ, VOCAB]
    loss   = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    print(f"Forward pass  OK | loss = {loss.item():.4f}")

    # ── Backward ─────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()

    # Sanity-check: at least one LoRA gradient is non-zero
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for m in model.singularis.decoder.modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A.weight, m.lora_B.weight]
    )
    print(f"Backward pass OK | LoRA gradients non-zero: {any_grad}")

    optimizer.step()
    print("Optimizer step OK\n")
    print("=== LoRA smoke test passed ===")
