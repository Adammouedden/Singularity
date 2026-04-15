from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2Config
from transformers.models.t5gemma2.modular_t5gemma2 import T5Gemma2Model, T5Gemma2RotaryEmbedding, T5Gemma2Encoder, T5Gemma2Decoder
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
import json
from safetensors.torch import load_file
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder, T5Gemma2Decoder
import torch

state_dict = load_file("reasoning/model.safetensors")
#print(state_dict)


with open("libs/transformers/src/transformers/models/t5gemma2/config.json") as f:
    data = json.load(f)

config = T5Gemma2Config.from_dict(data)


encoder = T5Gemma2Encoder(config.encoder)
decoder = T5Gemma2Decoder(config.decoder)

encoder_sd = {k[len("model.encoder."):]: v for k, v in state_dict.items() if k.startswith("model.encoder.")}
decoder_sd = {k[len("model.decoder."):]: v for k, v in state_dict.items() if k.startswith("model.decoder.")}

encoder.load_state_dict(encoder_sd, strict=False)
decoder.load_state_dict(decoder_sd, strict=False)

decoder = decoder.to(torch.bfloat16)
encoder = encoder.to(torch.bfloat16)

if __name__ == "__main__":

    # From config:
    # vocab_size: 262144  (boi=255999, eoi=256000, image=256001 — stay below these for text-only)
    # hidden_size: 640   (encoder output dim, decoder expects this)
    # image_size: 896, num_channels: 3, patch_size: 14  (for vision)

    BATCH      = 1
    ENC_SEQ    = 32   # encoder input token length
    DEC_SEQ    = 8    # decoder input token length (shorter — decoding step)
    VOCAB      = 255999  # safe upper bound (below special tokens)
    HIDDEN     = 640


    enc_input_ids    = torch.randint(0, VOCAB, (BATCH, ENC_SEQ), dtype=torch.int32)
    enc_attention_mask = torch.ones(BATCH, ENC_SEQ, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(BATCH, ENC_SEQ, HIDDEN, dtype=torch.bfloat16)


    pixel_values = torch.randn(BATCH, 3, 896, 896)


    dec_input_ids         = torch.randint(0, VOCAB, (BATCH, DEC_SEQ), dtype=torch.int32)
    dec_attention_mask    = torch.ones(BATCH, DEC_SEQ, dtype=torch.float)
    encoder_attention_mask = enc_attention_mask  # same shape as encoder input


    enc_out = encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
    print(enc_out.last_hidden_state)
    print("Encoder output complete. Now decoding...")


    # --- Run decoder with real encoder output ---
    dec_out = decoder(
        input_ids=dec_input_ids,
        attention_mask=dec_attention_mask,
        encoder_hidden_states=enc_out.last_hidden_state,
        encoder_attention_mask=enc_attention_mask,
    )
    print(dec_out)
    #dec_out.last_hidden_state shape: [BATCH, DEC_SEQ, HIDDEN]
