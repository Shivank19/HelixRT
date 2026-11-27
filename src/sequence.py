import numpy as np
import torch

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

VALID_BASES = set("ACGTN")

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}

def normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("DNA sequence must be a string.")

    sequence = sequence.upper()

    if not sequence:
        raise ValueError("DNA sequence cannot be empty.")

    if not set(sequence).issubset(VALID_BASES):
        raise ValueError("Invalid nucleotide in sequence.")

    return sequence

def reverse_complement(seq: str) -> str:
    seq = normalize_sequence(seq)
    return seq.translate(COMPLEMENT)[::-1]  

def canonicalize_sequence(seq: str) -> str:
    seq = normalize_sequence(seq)
    rc = reverse_complement(seq)
    return min(seq, rc)

def one_hot_encode(seq: str) -> np.ndarray:
    seq = normalize_sequence(seq)

    encoded = np.zeros((4, len(seq)), dtype=np.float32)

    for position, base in enumerate(seq):
        if base in BASE_TO_INDEX:
            channel = BASE_TO_INDEX[base]
            encoded[channel, position] = 1.0

    return encoded

def reverse_complement_tensor(x: torch.Tensor) -> torch.Tensor:
    # The tensor operation avoids decoding strings, transferring data back to the CPU and re-encoding sequences during model execution.
    
    if x.ndim not in (2, 3) or x.shape[-2] != 4:
        raise ValueError("Input tensor must have shape (4, L) or (B, 4, L).")
    
    x = x[...,[3, 2, 1, 0], :]
    return torch.flip(x, dims=[-1])