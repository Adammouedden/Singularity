import torch
from urm.URM import URM, URMCarry

device = torch.device("cuda" if torch.cuda_is.available() else "cpu")

#Replicated config from the URM paper
config_dict = {
    "batch_size": 2,
    "seq_len": 16,
    "puzzle_emb_ndim": 0,
    "num_puzzle_identifiers":10,
    "vocab_size": 32,
    "num_layers": 4,
    "hidden_size": 512,
    "expansion": 4.0,
    "num_heads": 8,
    "pos_encodings":"rope",
    "loops":16,
    "L_cycles":8,
    "H_cycles":1,
}

#Dummy input, randomly generated to test compilation and runtime
B, S = config_dict["batch_size"], config_dict["seq_len"]

model = URM(config_dict).to(device).eval()

batch = {
    "inputs":torch.randint(0, 32, (B,S), device=device),
    "puzzle_identifiers":torch.randint(0, 10, (B,), device=device),
}

carry = URMCarry(
    current_hidden=torch.zeros(B,S, config_dict["hidden_size"], dtype=torch.bfloat16, device=device),
    steps=torch.zeros(B, dtype=torch.int32, device=device),
    halted=torch.ones(B, dtype=torch.bool, device=device),
    current_data={k: v.clone() for k, v in batch.items()},
)

with torch.no_grad():
    new_carry, outputs = model(carry, batch)

print(new_carry)
print(outputs)
print("logits shape: ", outputs["logits"].shape)
print("q_halt_logits:", outputs["q_halt_logits"].shape)
