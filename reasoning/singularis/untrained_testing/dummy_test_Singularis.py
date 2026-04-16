"""
Smoke test: encoder → URM → decoder full pipeline.
Uses tiny randomly-initialized weights — output will be nonsense, but every
tensor shape and connection through the pipeline is verified with prints.

Run from the reasoning/ directory:
    python test_t5gemma2_urm.py
"""

import torch
from transformers.models.t5gemma2.configuration_t5gemma2 import (T5Gemma2Config, T5Gemma2TextConfig, T5Gemma2EncoderConfig, T5Gemma2DecoderConfig)
from transformers.models.siglip import SiglipVisionConfig
from reasoning.urm.URM import URMConfig
from reasoning.singularis.model import SingularisForConditionalGeneration

# ── Dimensions ───────────────────────────────────────────────────────────────
# H must be identical for encoder, decoder, AND URM — no projections.
H      = 640
VOCAB  = 262144
B      = 2    # batch size
S_ENC  = 8    # encoder sequence length
S_DEC  = 4    # decoder sequence length

# ── Build tiny T5Gemma2 config ────────────────────────────────────────────────
print("\n[1/4] Building T5Gemma2 config...")

text_cfg = T5Gemma2TextConfig(
    vocab_size=VOCAB,
    hidden_size=H,
    intermediate_size=H * 2,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=H // 2,
    max_position_embeddings=64,
)
# Minimal vision config — unused in this test (no pixel_values passed)
vision_cfg = SiglipVisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
    image_size=4,
    patch_size=2,
    num_channels=3,
)
enc_cfg = T5Gemma2EncoderConfig(text_config=text_cfg, vision_config=vision_cfg)
dec_cfg = T5Gemma2DecoderConfig(
    vocab_size=VOCAB,
    hidden_size=H,
    intermediate_size=H * 2,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=H // 2,
    max_position_embeddings=64,
)
t5_cfg = T5Gemma2Config(encoder=enc_cfg, decoder=dec_cfg)
print(f"    encoder hidden_size : {t5_cfg.encoder.text_config.hidden_size}")
print(f"    decoder hidden_size : {t5_cfg.decoder.hidden_size}")

# ── Build URM config ──────────────────────────────────────────────────────────
print("\n[2/4] Building URM config...")

urm_cfg = URMConfig(
    batch_size=B,
    seq_len=64,              # max RoPE positions — actual seq is sliced at runtime
    puzzle_emb_ndim=0,       # no puzzle embeddings
    num_puzzle_identifiers=1,
    vocab_size=1,            # unused: we bypass the URM's embedding table
    num_layers=2,
    hidden_size=H,           # MUST match T5Gemma2 hidden_size
    expansion=2.0,
    num_heads=4,
    pos_encodings="rope",
    loops=2,
    L_cycles=1,
    H_cycles=1,
)
print(f"    hidden_size : {urm_cfg.hidden_size}  (matches T5: {urm_cfg.hidden_size == H})")
print(f"    loops       : {urm_cfg.loops}, L_cycles={urm_cfg.L_cycles}")

# ── Instantiate model ─────────────────────────────────────────────────────────
print("\n[3/4] Building SingularisForConditionalGeneration...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    device : {device}")

model = SingularisForConditionalGeneration(t5_cfg, urm_cfg).to(device).eval()

total_params = sum(p.numel() for p in model.parameters())
enc_params   = sum(p.numel() for p in model.model.encoder.parameters())
urm_params   = sum(p.numel() for p in model.model.urm_bridge.parameters())
dec_params   = sum(p.numel() for p in model.model.decoder.parameters())
print(f"    total params      : {total_params:,}")
print(f"    encoder params    : {enc_params:,}")
print(f"    urm_bridge params : {urm_params:,}")
print(f"    decoder params    : {dec_params:,}")

# Verify the URM bridge is actually wired in
assert hasattr(model.model, "urm_bridge"), "urm_bridge not found on model.model!"
print("    urm_bridge present: True")

# ── Forward pass ──────────────────────────────────────────────────────────────
print(f"\n[4/4] Running forward pass  (B={B}, S_enc={S_ENC}, S_dec={S_DEC})...")

input_ids         = torch.randint(0, VOCAB, (B, S_ENC), device=device)
decoder_input_ids = torch.randint(0, VOCAB, (B, S_DEC), device=device)
attention_mask    = torch.ones(B, S_ENC, dtype=torch.long, device=device)

print(f"    input_ids         : {tuple(input_ids.shape)}")
print(f"    decoder_input_ids : {tuple(decoder_input_ids.shape)}")
print(f"    attention_mask    : {tuple(attention_mask.shape)}")

with torch.no_grad():
    # ── Encoder only (stage check) ────────────────────────────────────────────
    print("\n    --- stage 1: encoder ---")
    enc_out = model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    print(f"    encoder output shape   : {tuple(enc_out.last_hidden_state.shape)}")

    # ── URM bridge only (stage check) ────────────────────────────────────────
    print("\n    --- stage 2: URM bridge ---")
    enc_hidden = enc_out.last_hidden_state
    urm_out    = model.model.urm_bridge(enc_hidden)
    print(f"    urm input  shape       : {tuple(enc_hidden.shape)}")
    print(f"    urm output shape       : {tuple(urm_out.shape)}")
    assert urm_out.shape == enc_hidden.shape, \
        f"URM changed shape! {enc_hidden.shape} → {urm_out.shape}"
    print(f"    shape preserved        : True")
    diff = (urm_out - enc_hidden).abs().mean().item()
    print(f"    mean |urm_out - enc_in|: {diff:.6f}  (non-zero = URM did something)")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\n    --- stage 3: full pipeline ---")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    print(f"    logits shape                  : {tuple(outputs.logits.shape)}")
    print(f"    encoder_last_hidden_state     : {tuple(outputs.encoder_last_hidden_state.shape)}")
    assert outputs.logits.shape == (B, S_DEC, VOCAB), \
        f"Unexpected logits shape: {outputs.logits.shape}"

print("\n" + "=" * 60)
print("SUCCESS — full pipeline encoder → URM → decoder completed.")
print("Output is random (untrained weights) but all shapes are correct.")
print("=" * 60 + "\n")
