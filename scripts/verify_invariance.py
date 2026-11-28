from pathlib import Path

import torch

from src.models import build_model

from src.sequence import (
    reverse_complement,
    one_hot_encode,
    reverse_complement_tensor
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "invariant_seed42_epochs100.pt"
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model = build_model("invariant")

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


sequences = [
    "ATGCCA" * 83 + "AT",
    "CGTAAC" * 83 + "CG",
    "A" * 500,
    "G" * 500,
    "ACGT" * 125
]

predictions = []

with torch.inference_mode():

    for seq in sequences:

        rc = reverse_complement(seq)

        x = torch.from_numpy(
            one_hot_encode(seq)
        ).unsqueeze(0)

        x_rc = torch.from_numpy(
            one_hot_encode(rc)
        ).unsqueeze(0)

        # Independently verify the tensor transformation
        assert torch.equal(
            x_rc,
            reverse_complement_tensor(x)
        )

        # Separate model invocations, without caching
        p_original = torch.sigmoid(
            model(x)
        ).item()

        p_rc = torch.sigmoid(
            model(x_rc)
        ).item()

        error = abs(p_original - p_rc)

        predictions.append(p_original)

        print(f"\nOriginal: {seq[:30]}...")
        print(f"RC:       {rc[:30]}...")
        print(f"Identical inputs: {torch.equal(x, x_rc)}")

        print(f"Original prediction: {p_original:.10f}")
        print(f"RC prediction:       {p_rc:.10f}")
        print(f"Absolute error:      {error:.10e}")

        assert error < 1e-6


assert max(predictions) - min(predictions) > 1e-6

print("\nPrediction range:",
      min(predictions), max(predictions))

print("All sanity checks passed.")