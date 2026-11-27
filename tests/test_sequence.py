import numpy as np
import pytest
import torch

from src.sequence import (
    reverse_complement,
    canonicalize_sequence,
    one_hot_encode,
    reverse_complement_tensor
)

@pytest.mark.parametrize("sequence, expected", [
    ("ATGCCA", "TGGCAT"),
        ("AAAA", "TTTT"),
        ("ACGT", "ACGT"),
        ("ACNGT", "ACNGT"),
        ("atgcca", "TGGCAT")
])
def test_reverse_complement(sequence, expected):
    assert reverse_complement(sequence) == expected

@pytest.mark.parametrize("sequence", ["ATGCCA", "AAAA", "ACGT", "ACNGT"])
def test_reverse_complement_involution(sequence):
    assert reverse_complement(reverse_complement(sequence)) == sequence

def test_canonicalize_sequence():
    sequence = "ATGCCA"
    assert canonicalize_sequence(sequence) == canonicalize_sequence(reverse_complement(sequence))

def test_canonical_idempotence():
    sequence = "TGGCAT"
    assert canonicalize_sequence(canonicalize_sequence(sequence)) == canonicalize_sequence(sequence)

def test_one_hot_encoding():
    encoded = one_hot_encode("ACGT")
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_array_equal(encoded, expected)

def test_ambiguous_nucleotide():
    encoded = one_hot_encode("N")
    assert encoded.shape == (4, 1)
    assert encoded.sum() == 0

def test_tensor_reverse_complement():
    sequence = "ATGCCA"

    original = torch.from_numpy(one_hot_encode(sequence))

    expected = torch.from_numpy(one_hot_encode(reverse_complement(sequence)))

    actual = reverse_complement_tensor(original)

    assert torch.equal(actual, expected)

def test_batch_reverse_complement():

    original = torch.stack([
        torch.from_numpy(one_hot_encode("ATGCCA")),
        torch.from_numpy(one_hot_encode("AAAAAA"))
    ])

    transformed = reverse_complement_tensor(original)

    assert transformed.shape == original.shape

    assert torch.equal(reverse_complement_tensor(transformed), original)

@pytest.mark.parametrize("sequence", ["", "ABCD", "AT GX"])
def test_invalid_sequences(sequence):
    with pytest.raises(ValueError):
        reverse_complement(sequence)