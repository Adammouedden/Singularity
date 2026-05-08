import torch
import torch.nn as nn
from reasoning.urm.URM import URM, URMCarry, URMConfig


class URMBridge(nn.Module):
    """
    Bridges T5Gemma2 encoder output → URM → T5Gemma2 decoder input.

    The URM hidden_size must equal the encoder/decoder hidden_size.
    No projection layers are used — encoder hidden states are fed directly
    into the URM as the input signal at each reasoning cycle, bypassing
    the URM's internal token embedding table.

    Data flow per forward pass:
        encoder_hidden_states [B, S, H]
            → URM (N loops of: hidden += encoder_states, then transformer blocks)
            → refined_hidden_states [B, S, H]   (same shape, passed to decoder)
    """

    def __init__(self, urm_config: URMConfig):
        super().__init__()
        self.urm = URM(urm_config.model_dump())
        self.urm_hidden_size = urm_config.hidden_size
        self.forward_dtype = getattr(torch, urm_config.forward_dtype)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [B, S, H]  (H must equal urm_config.hidden_size)
        Returns:
            refined_hidden_states: [B, S, H]  (same shape — drop-in for decoder cross-attn)
        """
        B, S, H = encoder_hidden_states.shape
        device = encoder_hidden_states.device

        assert H == self.urm_hidden_size, (
            f"Encoder hidden size ({H}) must equal URMConfig.hidden_size ({self.urm_hidden_size})."
        )

        # Initialise carry: zero memory state, all sequences marked as "halted"
        # (halted=True triggers a carry reset inside URM.forward — correct for step 0)
        carry = URMCarry(
            current_hidden=torch.zeros(
                B, S, self.urm_hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            steps=torch.zeros(B, dtype=torch.int32, device=device),
            halted=torch.ones(B, dtype=torch.bool, device=device),
            current_data={},
        )

        # Run URM — encoder output is the input signal injected each cycle.
        # batch={} because we bypass the URM's internal token embedding table.
        new_carry, _ = self.urm(
            carry,
            batch={},
            input_hidden_states=encoder_hidden_states,
        )

        # Return the final reasoned hidden state, cast back to the encoder's dtype
        return new_carry.current_hidden.to(encoder_hidden_states.dtype)
