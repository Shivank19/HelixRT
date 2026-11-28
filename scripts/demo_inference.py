from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import InferenceEngine

from src.sequence import reverse_complement


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    seq_a = test_df.iloc[0]["seq"]

    seq_b = test_df.iloc[-1]["seq"]

    rc_a = reverse_complement(seq_a)

    rc_b = reverse_complement(seq_b)

    requests = [
        seq_a,
        seq_b,
        rc_a,
        seq_a,
        rc_b,
        seq_b
    ]

    modes = [
        "none",
        "exact",
        "canonical",
        "canonical_dedup",
        "dedup_only"
    ]

    baseline_predictions = None

    for mode in modes:

        print("\n" + "=" * 60)

        print("Inference mode:", mode)

        engine = InferenceEngine.from_checkpoint(
            model_name="invariant",
            seed=42,
            epochs=100,
            device="cpu",
            mode=mode,
            cache_capacity=128,
            batch_size=128
        )

        first_predictions = engine.predict(requests)

        if baseline_predictions is None:

            baseline_predictions = (first_predictions)

        np.testing.assert_allclose(
            first_predictions,
            baseline_predictions,
            atol=1e-6,
            rtol=0
        )

        second_predictions = engine.predict(
            requests
        )

        np.testing.assert_allclose(
            second_predictions,
            baseline_predictions,
            atol=1e-6,
            rtol=0
        )

        print("\nPredictions:")

        for index, probability in enumerate(
            first_predictions
        ):

            print(
                f"Request {index}: "
                f"{probability:.8f}"
            )

        print("\nStatistics:")

        for key, value in engine.stats().items():

            print(f"{key}: {value}")

    print("\nAll inference modes passed.")


if __name__ == "__main__":

    main()