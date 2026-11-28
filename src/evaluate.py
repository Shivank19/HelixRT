import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss

from src.dataset import get_test_loader
from src.models import build_model
from src.sequence import reverse_complement_tensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

MODELS = ["standard", "augmented", "invariant"]

SEEDS = [42, 123, 456]


def calculate_metrics(labels, probabilities, rc_probabilities):

    if len(labels) == 0:
        raise ValueError("No samples to evaluate.")

    if not (labels.shape == probabilities.shape == rc_probabilities.shape):
        raise ValueError("Prediction and label shapes do not match.")

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite original predictions detected.")

    if not np.isfinite(rc_probabilities).all():
        raise ValueError("Non-finite RC predictions detected.")
    
    predictions = (probabilities >= 0.5).astype(int)
    rc_predictions = (rc_probabilities >= 0.5).astype(int)

    rc_errors = np.abs(probabilities - rc_probabilities)

    return{
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auroc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "rc_mean_error": float(rc_errors.mean()),
        "rc_max_error": float(rc_errors.max()),
        "rc_flip_rate": float((predictions != rc_predictions).mean()),
        "rc_flip_count": int((predictions != rc_predictions).sum())
    }

@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    
    all_labels, all_probabilities, all_rc_probabilities = [], [], []

    for features, labels in loader:
        features = features.to(device)
        rc_features = reverse_complement_tensor(features)
        probabilities = torch.sigmoid(model(features))
        rc_probabilities = torch.sigmoid(model(rc_features))
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_rc_probabilities.append(rc_probabilities.cpu().numpy())

    return(
        np.concatenate(all_labels), 
        np.concatenate(all_probabilities), 
        np.concatenate(all_rc_probabilities)
    )

def load_checkpoint(model_name, seed, device):

    checkpoint_path = (CHECKPOINT_DIR / f"{model_name}_seed{seed}_epochs100.pt")

    if not checkpoint_path.exists():
         raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if checkpoint["model_name"] != model_name:

        raise ValueError("Checkpoint model name mismatch.")

    if checkpoint["seed"] != seed:

        raise ValueError("Checkpoint seed mismatch.")

    model = build_model(model_name)

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    model.eval()

    print(f"\nLoaded checkpoint: " f"{checkpoint_path.name}")

    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    print(f"Validation AUROC: " f"{checkpoint['val_auroc']:.4f}")

    return model, checkpoint

def record_results(model_name, seed, checkpoint_epoch, validation_auroc, labels, probabilities, rc_probabilities, metric_rows, prediction_frames):
    metrics = calculate_metrics(labels, probabilities, rc_probabilities)

    metric_rows.append({

        "model": model_name,
        "seed": seed,
        "checkpoint_epoch": checkpoint_epoch,
        "validation_auroc": validation_auroc,

        **metrics
    })

    predictions = pd.DataFrame({

        "sequence_id": np.arange(len(labels)),
        "model": model_name,
        "seed": seed,
        "label": labels.astype(int),
        "probability": probabilities,
        "rc_probability": rc_probabilities,
        "prediction": (probabilities >= 0.5).astype(int),
        "rc_prediction": (rc_probabilities >= 0.5).astype(int),
        "rc_error": np.abs(probabilities - rc_probabilities),
        "rc_flip": ((probabilities >= 0.5) != (rc_probabilities >= 0.5))
    })

    prediction_frames.append(predictions)

    print(
        f"{model_name:20s} | "
        f"Seed: {str(seed):4s} | "
        f"AUROC: {metrics['auroc']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"RC Error: {metrics['rc_mean_error']:.8e} | "
        f"Flip Rate: {metrics['rc_flip_rate']:.4%}"
    )

def evaluate_gc_baseline(metric_rows, prediction_frames):
    train_df = pd.read_csv(DATA_DIR/"train.csv")
    test_df = pd.read_csv(DATA_DIR/"test.csv")

    model = LogisticRegression(random_state=42)
    model.fit(train_df[["gc_content"]], train_df["label"])

    labels = test_df["label"].to_numpy()

    probabilities = model.predict_proba(test_df[["gc_content"]])[:, 1]

    # Independently calculate GC content from the reverse-complement sequences.
    from src.sequence import reverse_complement

    def gc_content(sequence):
        sequence = sequence.upper()
        called_bases = sum(sequence.count(base) for base in "ACGT")

        if called_bases == 0:
            raise ValueError("Sequence contains no called bases")

        return (sequence.count("G") + sequence.count("C")) / called_bases

    rc_gc = test_df["seq"].apply(lambda seq: gc_content(reverse_complement(seq)))

    rc_features = pd.DataFrame({
        "gc_content": rc_gc
    })

    rc_probabilities = model.predict_proba(rc_features)[:, 1]

    record_results(
        model_name="gc_baseline",
        seed=42,
        checkpoint_epoch=None,
        validation_auroc=None,
        labels=labels,
        probabilities=probabilities,
        rc_probabilities=rc_probabilities,
        metric_rows=metric_rows,
        prediction_frames=prediction_frames
    )

def evaluate_all(args):

    device = torch.device(args.device)
    if (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Evaluation device:", device)

    test_loader = get_test_loader(batch_size=args.batch_size, num_workers=0)

    print("Test samples:", len(test_loader.dataset))

    metric_rows = []

    prediction_frames = []

    evaluate_gc_baseline(metric_rows, prediction_frames)

    for model_name in MODELS:
        for seed in SEEDS:
            model, checkpoint = load_checkpoint(model_name, seed, device)

            labels, probabilities, rc_probabilities = (predict(model, test_loader, device))

            record_results(

                model_name=model_name,
                seed=seed,
                checkpoint_epoch=checkpoint["epoch"],
                validation_auroc=checkpoint["val_auroc"],
                labels=labels,
                probabilities=probabilities,
                rc_probabilities=rc_probabilities,
                metric_rows=metric_rows,
                prediction_frames=prediction_frames
            )

            if model_name == "invariant":
                np.testing.assert_allclose(probabilities, rc_probabilities, atol=1e-6, rtol=0)

            if model_name in ("standard", "augmented"):
                tta_probabilities = (probabilities + rc_probabilities) / 2.0

                # Reversing the input orientation exchanges the two operands.

                tta_rc_probabilities = (rc_probabilities + probabilities) / 2.0

                record_results(
                    model_name=f"{model_name}_tta",
                    seed=seed,
                    checkpoint_epoch=checkpoint["epoch"],
                    validation_auroc=None,
                    labels=labels,
                    probabilities=tta_probabilities,
                    rc_probabilities=tta_rc_probabilities,
                    metric_rows=metric_rows,
                    prediction_frames=prediction_frames
                )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    expected_runs = 1 + 5 * len(SEEDS)

    expected_rows = (expected_runs * len(test_loader.dataset))

    assert len(metrics_df) == expected_runs, (f"Expected {expected_runs} evaluations, "         f"got {len(metrics_df)}")

    assert len(predictions_df) == expected_rows, (f"Expected {expected_rows} prediction rows, " f"got {len(predictions_df)}")

    assert metrics_df["auroc"].notna().all(), ("Undefined AUROC detected.")

    print("\nEvaluation integrity checks passed.")

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "average_precision",
        "log_loss",
        "rc_mean_error",
        "rc_max_error",
        "rc_flip_rate",
        "rc_flip_count"
    ]

    grouped = metrics_df.groupby("model")
    summary_df = grouped[metric_columns].agg(["mean", "std"])

    summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in summary_df.columns]

    summary_df["n_runs"] = grouped.size()

    summary_df = summary_df.reset_index()

    summary_df = summary_df.sort_values("auroc_mean", ascending=False)

    metrics_df.to_csv(RESULTS_DIR / "test_metrics.csv",index=False)

    predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv",index=False)

    summary_df.to_csv(RESULTS_DIR / "model_comparison.csv",index=False)

    print("\nAggregated test results:")

    print(summary_df[[
            "model",
            "n_runs",
            "auroc_mean",
            "auroc_std",
            "accuracy_mean",
            "f1_mean",
            "rc_mean_error_mean",
            "rc_flip_rate_mean"
        ]].to_string(index=False)
    )

    print("\nEvaluation completed.")

    print("Saved metrics to:", RESULTS_DIR)

    print("Prediction rows:", len(predictions_df))


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--batch-size", type=int, default=128)

    return parser.parse_args()


if __name__ == "__main__":

    evaluate_all(parse_args())