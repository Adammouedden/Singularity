import torch
from reasoning.urm.URM import URMConfig
from reasoning.singularis.model import SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import config   # also loads encoder + decoder as side effect
from transformers import AutoTokenizer

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"   # fix: was torch.is_available()

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# T5Gemma2 uses the PaliGemma2 tokenizer (SentencePiece, vocab=262144)
tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-448")

# ── URM config (must match hidden_size=640 from T5Gemma2 config) ──────────────
urm_cfg = URMConfig(
    batch_size=1,
    seq_len=512,
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=1,
    vocab_size=1,           # unused — we bypass URM's embedding table
    num_layers=2,
    hidden_size=640,        # must match T5Gemma2 hidden_size
    expansion=2.0,
    num_heads=8,            # head_dim = 640//8 = 80, safe for FlashAttention
    pos_encodings="rope",
    loops=2,
    L_cycles=1,
    H_cycles=1,
)

# ── Build model ───────────────────────────────────────────────────────────────
# config is already loaded from config.json by config_and_weights import.
# encoder + decoder are already loaded with safetensors weights (same import).
# tie_weights() inside __init__ ties lm_head.weight → decoder.embed_tokens.weight.
model = SingularisForConditionalGeneration(config, urm_cfg).to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"Prompt : {prompt!r}")
print(f"Tokens : {inputs['input_ids'].shape}")

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        do_sample=False,    # greedy decoding
    )

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Output : {response!r}")
