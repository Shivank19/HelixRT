import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score)

from src.dataset import get_dataloaders
from src.models import build_model
from src.sequence import reverse_complement_tensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_devide():
    if torch.cuda.is_available():

        return torch.device("cuda")

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


def augment_batch(x):
    mask = torch.rand(x.shape[0], device = x.device) < 0.5

    rc = reverse_complement_tensor(x)

    return torch.where(mask[:, None, None], rc, x)

def train_one_epoch(model, loader, optimizer, criterion, device, augment=False):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if augment:
            features = augment_batch(features)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += (loss.item() * labels.size(0))

    return total_loss / len(loader.dataset)

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_probabilities = []
    all_rc_probabilities = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits)

        rc_features = reverse_complement_tensor(features)

        rc_logits = model(rc_features)

        rc_probabilities = torch.sigmoid(rc_logits)

        total_loss += (loss.item() * labels.size(0))

        all_labels.append(labels.cpu().numpy())

        all_probabilities.append(probabilities.cpu().numpy())

        all_rc_probabilities.append(rc_probabilities.cpu().numpy())

    labels = np.concatenate(all_labels)

    probabilities = np.concatenate(all_probabilities)

    rc_probabilities = np.concatenate(all_rc_probabilities)

    predictions = (probabilities >= 0.5).astype(int)

    rc_predictions = (rc_probabilities >= 0.5).astype(int)

    rc_errors = np.abs(probabilities - rc_probabilities)

    return {

        "loss": total_loss / len(loader.dataset),

        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),

        "auroc": roc_auc_score(
            labels,
            probabilities
        ),

        "average_precision": average_precision_score(
            labels,
            probabilities
        ),

        "rc_mean_error": rc_errors.mean(),

        "rc_max_error": rc_errors.max(),

        "rc_flip_rate": (
            predictions != rc_predictions
        ).mean()
    }

def get_device():

    
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed = args.seed,
        pin_memory = (device.type == "cuda")
    )

    model = build_model(args.model).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.learning_rate,
        weight_decay = 1e-4
    )

    use_augmentation = (args.model == "augmented")

    best_auroc = -float("inf")

    epochs_without_improvement = 0

    history = []

    run_name = (
        f"{args.model}_seed{args.seed}"
        f"_epochs{args.epochs}"
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"{run_name}.pt"
    )

    history_path = (
        RESULTS_DIR / f"{run_name}_history.csv"
    )

    print("\nModel:", args.model)
    print("Augmentation:", use_augmentation)
    print("Epoch budget:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            augment=use_augmentation
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        elapsed = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_average_precision": val_metrics["average_precision"],
            "rc_mean_error": val_metrics["rc_mean_error"],
            "rc_max_error": val_metrics["rc_max_error"],
            "rc_flip_rate": val_metrics["rc_flip_rate"],
            "epoch_seconds": elapsed
        }

        history.append(row)

        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | "
            f"RC Error: "
            f"{val_metrics['rc_mean_error']:.8f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_auroc": best_auroc,
                    "config": vars(args)
                },
                checkpoint_path
            )
            print("  Saved best checkpoint.")

        else:

            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement
            >= args.patience
        ):

            print("\nEarly stopping triggered.")

            break

    print("\nTraining completed.")

    print(
        f"Best validation AUROC: "
        f"{best_auroc:.4f}"
    )

    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "standard",
            "augmented",
            "invariant"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train(args)

    