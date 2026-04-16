import torch
from reasoning.urm.URM import URMConfig
from reasoning.singularis.model import SingularisForConditionalGeneration
from transformers import AutoTokenizer
from reasoning.singularis.config_and_weights import LLM_config, encoder_weights, decoder_weights, URM_config

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu" 

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# T5Gemma2 uses the PaliGemma2 tokenizer (SentencePiece, vocab=262144)
tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-448")

# ── Build model ───────────────────────────────────────────────────────────────
# config is already loaded from config.json by config_and_weights import.
# encoder + decoder are already loaded with safetensors weights (same import).
# tie_weights() inside __init__ ties lm_head.weight → decoder.embed_tokens.weight.
model = SingularisForConditionalGeneration(URM_config, LLM_config).to(device).eval()
model.load_weights(encoder_weights, decoder_weights)

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
