from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from src.sequence import one_hot_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEQUENCE_LENGTH = 500

# print("Project root:", PROJECT_ROOT)
# print("Data directory:", DATA_DIR)

# print("Train exists:", (DATA_DIR / "train.csv").exists())
# print("Val exists:", (DATA_DIR / "val.csv").exists())
# print("Test exists:", (DATA_DIR / "test.csv").exists())

class GenomicDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.sequences = self.df["seq"].tolist()
        self.labels = torch.tensor(self.df["label"].to_numpy(), dtype=torch.float32)

        if not self.df["label"].isin([0, 1]).all():
            raise ValueError("Labels must be either 0 or 1")

        if not all(len(seq) == SEQUENCE_LENGTH for seq in self.sequences):
            raise ValueError(f"All sequences must have length {SEQUENCE_LENGTH}")

        encoded = np.stack([one_hot_encode(seq) for seq in self.sequences])
        self.features = torch.from_numpy(encoded)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloaders(batch_size=128, num_workers=0, seed=42, pin_memory=False):
    
    train_dataset = GenomicDataset(DATA_DIR / "train.csv")

    val_dataset = GenomicDataset(DATA_DIR / "val.csv")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

def get_test_loader(batch_size=128, num_workers=0, pin_memory=False):

    test_dataset = GenomicDataset(DATA_DIR / "test.csv")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return test_loader

if __name__ == "__main__":

    train_loader, val_loader = get_dataloaders()
    features, labels = next(iter(train_loader))

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    print("Input batch shape:", features.shape)
    print("Label batch shape:", labels.shape)

    print("Input dtype:", features.dtype)
    print("Label dtype:", labels.dtype) 