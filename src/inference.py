from pathlib import Path

import numpy as np
import torch

from src.cache import LRUCache
from src.models import build_model

from src.sequence import (normalize_sequence, canonicalize_sequence, one_hot_encode, reverse_complement_tensor)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

SEQUENCE_LENGTH = 500

VALID_MODES = {
    "none",
    "exact",
    "canonical",
    "canonical_dedup",
    "dedup_only"
} 

class InferenceEngine:
    def __init__(self, model, model_name, device="cpu", mode="none", cache_capacity=4096, batch_size=128, tta=False):

        if model_name not in {"standard", "augmented", "invariant"}:
            raise ValueError(f"Unknown model: {model_name}")

        if mode not in VALID_MODES:
            raise ValueError(f"Invalid inference mode: {mode}")

        if (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("Batch size must be a positive integer.")

        self.device = torch.device(device)

        if (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("CUDA is unavailable.")

        self.model_name = model_name
        self.mode = mode
        self.batch_size = batch_size
        self.tta = tta

        if model_name == "invariant" and tta:
            raise ValueError("TTA is unnecessary for the invariant model.")

        self.is_invariant = (model_name == "invariant" or tta)

        self.use_canonical = mode in {"canonical", "canonical_dedup", "dedup_only"}

        self.use_dedup = mode in {"canonical_dedup", "dedup_only"}

        self.use_cache = mode in {"exact", "canonical", "canonical_dedup"}

        if (self.use_canonical and not self.is_invariant):
            raise ValueError("Canonical inference requires a strand-invariant model.")

        self.model = model.to(self.device)

        self.model.eval()

        self.cache = (LRUCache(cache_capacity) if self.use_cache else None)

        self.reset_stats()

    @classmethod
    def from_checkpoint(cls, model_name, seed=42, epochs=100, **kwargs):

        checkpoint_path = (CHECKPOINT_DIR / f"{model_name}_seed{seed}_epochs{epochs}.pt")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if checkpoint["model_name"] != model_name:
            raise ValueError("Checkpoint model mismatch.")

        if checkpoint["seed"] != seed:
            raise ValueError("Checkpoint seed mismatch.")

        model = build_model(model_name)

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        return cls(model, model_name=model_name, **kwargs)

    def reset_stats(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_duplicates = 0
        self.model_evaluations = 0
        self.forward_calls = 0
        if self.cache is not None:
            self.cache.reset_stats()

    def clear_cache(self):
        if self.cache is not None:
            self.cache.clear()
        self.reset_stats()

    def _validate_sequence(self, sequence):

        sequence = normalize_sequence(sequence)
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(f"Expected {SEQUENCE_LENGTH} bases, received {len(sequence)}.")
        return sequence

    def _make_key(self, sequence):
        if self.use_canonical:
            return canonicalize_sequence(sequence)
        return sequence

    @torch.inference_mode()
    def _run_model(self, sequences):
        if not sequences:
            return []

        outputs = []

        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start:start + self.batch_size]

            encoded = np.stack([one_hot_encode(sequence) for sequence in batch])
            x = torch.from_numpy(encoded)
            x = x.to(self.device)

            if self.tta:
                rc = reverse_complement_tensor(x)
                combined = torch.cat([x, rc], dim=0)

                probabilities = torch.sigmoid(self.model(combined))

                p_forward, p_rc = probabilities.chunk(2, dim=0)

                probabilities = (p_forward + p_rc) / 2.0

            else:
                probabilities = torch.sigmoid(self.model(x))

            if probabilities.shape != (len(batch),):
                raise RuntimeError(f"Unexpected output shape: {probabilities.shape}")

            batch_outputs = probabilities.cpu().tolist()

            if not all(np.isfinite(p) and 0 <= p <= 1 for p in batch_outputs):
                raise RuntimeError("Invalid model predictions.")

            outputs.extend(batch_outputs)

            self.model_evaluations += len(batch)

            self.forward_calls += 1

        return outputs
    
    def predict(self, sequences):

        if isinstance(sequences, str):
            raise TypeError("Expected a collection of DNA sequences.")

        sequences = list(sequences)

        if not sequences:
            return []

        # Validate the entire request before modifying cache contents or statistics.

        normalized = [self._validate_sequence(sequence)for sequence in sequences]

        keys = [self._make_key(sequence) for sequence in normalized]

        self.total_requests += len(sequences)

        results = [None] * len(sequences)

        pending_keys = []
        pending_indices = []
        pending_lookup = {}

        for index, key in enumerate(keys):

            if self.use_cache:
                cached = self.cache.get(key)

                if cached is not None:
                    self.cache_hits += 1
                    results[index] = cached
                    continue

                self.cache_misses += 1

            if (self.use_dedup and key in pending_lookup):

                position = pending_lookup[key]
                pending_indices[position].append(index)
                self.batch_duplicates += 1
                continue

            position = len(pending_keys)
            pending_keys.append(key)
            pending_indices.append([index])

            if self.use_dedup:
                pending_lookup[key] = position

        predictions = self._run_model(pending_keys)

        for key, indices, probability in zip(pending_keys, pending_indices, predictions):

            if self.use_cache:
                self.cache.put(key, probability)

            for index in indices:
                results[index] = probability

        if any(p is None for p in results):
            raise RuntimeError("Incomplete inference results.")

        return results

    def stats(self):

        encoder_multiplier = (2 if self.is_invariant else 1)

        avoided = (self.total_requests - self.model_evaluations)

        reduction = (avoided / self.total_requests if self.total_requests > 0 else 0.0)

        cache_stats = (self.cache.stats() if self.cache is not None else {})

        return {
            "model": self.model_name,
            "mode": self.mode,
            "tta": self.tta,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batch_duplicates": self.batch_duplicates,
            "model_evaluations": self.model_evaluations,
            "forward_calls": self.forward_calls,
            "encoder_evaluations": self.model_evaluations * encoder_multiplier,
            "model_evaluations_avoided": avoided,
            "evaluation_reduction": reduction,
            "cache": cache_stats
        }