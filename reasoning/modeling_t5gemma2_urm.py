import os
import sys

# reasoning/ is the directory containing this file — needed for urm_bridge and urm.URM imports
_REASONING_DIR = os.path.dirname(os.path.abspath(__file__))

# T5Gemma2 is cloned separately outside the repo: Singularity/T5Gemma2/transformers/src
_T5GEMMA2_SRC = os.path.abspath(
    os.path.join(_REASONING_DIR, "..", "..", "..", "T5Gemma2", "transformers", "src")
)

for _p in [_REASONING_DIR, _T5GEMMA2_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2Model,
    T5Gemma2ForConditionalGeneration,
)
from transformers.modeling_outputs import Seq2SeqModelOutput
from urm.URM import URMConfig
from urm_bridge import URMBridge
from scaffold_Singularis import encoder, decoder


class T5Gemma2WithURMModel(T5Gemma2Model):

    def __init__(self, config, urm_config: URMConfig):
        super().__init__(config)
        self.encoder = encoder
        self.urm_bridge = URMBridge(urm_config)
        self.decoder = decoder

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ) -> Seq2SeqModelOutput:

        # Step 1: Run encoder (or reuse a cached encoder_outputs from a prior call)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                return_dict=True,
                **kwargs,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, S_enc, H]

        # Step 2: Run URM — refines the encoder representation through N reasoning loops
        encoder_hidden_states = self.urm_bridge(encoder_hidden_states)  # [B, S_enc, H]

        # Step 3: Run decoder using the URM-refined states as cross-attention keys/values
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    


class T5Gemma2WithURMForConditionalGeneration(T5Gemma2ForConditionalGeneration):
    """
    Drop-in replacement for T5Gemma2ForConditionalGeneration with a URM
    reasoning module inserted between the encoder and decoder.
    """

    def __init__(self, config, urm_config: URMConfig):
        super().__init__(config)
        # Replace the base model with the URM-augmented version, then retie
        # weights so lm_head.out_proj points to the new model's embed_tokens
        # (super().__init__ tied them to the first temporary T5Gemma2Model).
        self.model = T5Gemma2WithURMModel(config, urm_config)
        self.tie_weights()
