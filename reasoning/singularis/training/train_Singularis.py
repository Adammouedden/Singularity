from reasoning.singularis.model import Singularis, SingularisForConditionalGeneration
from reasoning.singularis.config_and_weights import URM_config, LLM_config, encoder_weights, decoder_weights

model = SingularisForConditionalGeneration(URM_config, LLM_config)

