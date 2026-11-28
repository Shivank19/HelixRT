import copy

import numpy as np
import pytest
import torch

from src.inference import InferenceEngine

from src.models import (StandardCNN, InvariantCNN)

from src.sequence import (reverse_complement)


SEQ_A = "ATGCCA" * 83 + "AT"

SEQ_B = "CGTAAC" * 83 + "CG"

SEQ_C = "ACGT" * 125

RC_A = reverse_complement(SEQ_A)

RC_B = reverse_complement(SEQ_B)


@pytest.fixture
def invariant_model():

    torch.manual_seed(42)

    return InvariantCNN()


def make_engine(model, mode, **kwargs):

    return InferenceEngine(
        model=copy.deepcopy(model),
        model_name="invariant",
        device="cpu",
        mode=mode,
        **kwargs
    )


def test_empty_requests(invariant_model):

    engine = make_engine(invariant_model, "canonical_dedup")

    assert engine.predict([]) == []

    assert engine.stats()["total_requests"] == 0


@pytest.mark.parametrize(
    "sequence",
    [
        "",
        "ACGT",
        "X" * 500,
        "A" * 499,
        "A" * 501,
        None
    ]
)
def test_invalid_sequences(invariant_model, sequence):

    engine = make_engine(invariant_model, "canonical_dedup")

    with pytest.raises((ValueError, TypeError)):
        engine.predict([sequence])


def test_invalid_mode(invariant_model):

    with pytest.raises(ValueError):
        make_engine(invariant_model, "unknown")


def test_invalid_batch_size(invariant_model):

    with pytest.raises(ValueError):
        make_engine(invariant_model, "none", batch_size=0)


def test_reject_single_string(invariant_model):
    engine = make_engine(invariant_model,"none")

    with pytest.raises(TypeError):
        engine.predict(SEQ_A)


def test_prediction_shape(invariant_model):

    engine = make_engine(invariant_model, "none" )

    predictions = engine.predict([SEQ_A, SEQ_B, SEQ_C])

    assert len(predictions) == 3

    assert all(0 <= p <= 1 for p in predictions)


def test_uncached_invariance(invariant_model):

    engine = make_engine(
        invariant_model,
        "none"
    )

    original = engine.predict([SEQ_A])

    rc = engine.predict([RC_A])

    np.testing.assert_allclose(original, rc, atol=1e-6, rtol=0 )

    assert engine.stats()["model_evaluations" ] == 2


@pytest.mark.parametrize(
    "mode",
    [
        "exact",
        "canonical",
        "canonical_dedup",
        "dedup_only"
    ]
)
def test_optimized_prediction_correctness(invariant_model, mode):

    baseline = make_engine(invariant_model, "none")

    optimized = make_engine(invariant_model, mode)

    sequences = [
        SEQ_A,
        RC_B,
        SEQ_B,
        RC_A,
        SEQ_A,
        SEQ_C
    ]

    expected = baseline.predict(sequences)

    actual = optimized.predict(sequences)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=0
    )


def test_exact_cache(invariant_model):

    engine = make_engine(
        invariant_model,
        "exact"
    )

    first = engine.predict([SEQ_A])

    second = engine.predict([SEQ_A])

    assert first == second

    stats = engine.stats()

    assert stats["cache_hits"] == 1

    assert stats["model_evaluations"] == 1


def test_exact_cache_distinguishes_rc(
    invariant_model
):

    engine = make_engine(
        invariant_model,
        "exact"
    )

    engine.predict([SEQ_A])

    engine.predict([RC_A])

    stats = engine.stats()

    assert stats["cache_hits"] == 0

    assert stats["model_evaluations"] == 2

    assert len(engine.cache) == 2


def test_canonical_cache(invariant_model):

    engine = make_engine(
        invariant_model,
        "canonical"
    )

    original = engine.predict([SEQ_A])

    rc = engine.predict([RC_A])

    assert original == rc

    stats = engine.stats()

    assert stats["cache_hits"] == 1

    assert stats["model_evaluations"] == 1

    assert len(engine.cache) == 1


def test_batch_deduplication(invariant_model):

    engine = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    predictions = engine.predict([
        SEQ_A,
        RC_A,
        SEQ_B,
        RC_B,
        SEQ_A
    ])

    assert len(predictions) == 5

    assert predictions[0] == predictions[1]

    assert predictions[0] == predictions[4]

    assert predictions[2] == predictions[3]

    stats = engine.stats()

    assert stats["total_requests"] == 5

    assert stats["batch_duplicates"] == 3

    assert stats["model_evaluations"] == 2

    assert stats["encoder_evaluations"] == 4


def test_dedup_without_cache(invariant_model):

    engine = make_engine(
        invariant_model,
        "dedup_only"
    )

    engine.predict([SEQ_A, RC_A])

    engine.predict([SEQ_A, RC_A])

    stats = engine.stats()

    assert stats["cache_hits"] == 0

    assert stats["batch_duplicates"] == 2

    assert stats["model_evaluations"] == 2

    assert engine.cache is None


def test_cache_and_dedup_combined(
    invariant_model
):

    engine = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    engine.predict([SEQ_A, RC_A])

    engine.predict([
        RC_A,
        SEQ_B,
        RC_B
    ])

    stats = engine.stats()

    assert stats["total_requests"] == 5

    assert stats["cache_hits"] == 1

    assert stats["batch_duplicates"] == 2

    assert stats["model_evaluations"] == 2


def test_request_order_preserved(
    invariant_model
):

    baseline = make_engine(
        invariant_model,
        "none"
    )

    optimized = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    optimized.predict([SEQ_A])

    sequences = [
        SEQ_B,
        SEQ_A,
        RC_B,
        SEQ_C,
        RC_A
    ]

    expected = baseline.predict(sequences)

    actual = optimized.predict(sequences)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=0
    )


def test_model_batch_chunking(
    invariant_model
):

    engine = make_engine(
        invariant_model,
        "none",
        batch_size=2
    )

    engine.predict([
        SEQ_A,
        SEQ_B,
        SEQ_C,
        RC_A,
        RC_B
    ])

    stats = engine.stats()

    assert stats["model_evaluations"] == 5

    assert stats["forward_calls"] == 3


def test_cache_capacity(invariant_model):

    engine = make_engine(
        invariant_model,
        "canonical_dedup",
        cache_capacity=1
    )

    engine.predict([SEQ_A])

    engine.predict([SEQ_B])

    assert len(engine.cache) == 1

    assert engine.cache.evictions == 1

    engine.predict([RC_A])

    assert engine.stats()[
        "model_evaluations"
    ] == 3


def test_clear_cache(invariant_model):

    engine = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    engine.predict([SEQ_A])

    engine.clear_cache()

    assert len(engine.cache) == 0

    assert engine.stats()["total_requests"] == 0


def test_reset_stats_preserves_cache(
    invariant_model
):

    engine = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    engine.predict([SEQ_A])

    engine.reset_stats()

    engine.predict([RC_A])

    stats = engine.stats()

    assert stats["total_requests"] == 1

    assert stats["cache_hits"] == 1

    assert stats["model_evaluations"] == 0


@pytest.mark.parametrize(
    "mode",
    [
        "none",
        "exact",
        "canonical",
        "canonical_dedup",
        "dedup_only"
    ]
)
def test_request_accounting(
    invariant_model,
    mode
):

    engine = make_engine(
        invariant_model,
        mode
    )

    engine.predict([
        SEQ_A,
        RC_A,
        SEQ_B
    ])

    engine.predict([
        SEQ_A,
        RC_B,
        SEQ_C,
        SEQ_C
    ])

    stats = engine.stats()

    assert stats["total_requests"] == (
        stats["cache_hits"]
        + stats["batch_duplicates"]
        + stats["model_evaluations"]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "canonical",
        "canonical_dedup",
        "dedup_only"
    ]
)
def test_unsafe_canonicalization_rejected(mode):

    with pytest.raises(ValueError):

        InferenceEngine(
            model=StandardCNN(),
            model_name="standard",
            mode=mode,
            tta=False
        )


def test_tta_invariance():

    torch.manual_seed(42)

    engine = InferenceEngine(
        model=StandardCNN(),
        model_name="standard",
        mode="none",
        tta=True
    )

    original = engine.predict([SEQ_A])

    rc = engine.predict([RC_A])

    np.testing.assert_allclose(
        original,
        rc,
        atol=1e-6,
        rtol=0
    )


def test_tta_canonical_cache():

    engine = InferenceEngine(
        model=StandardCNN(),
        model_name="standard",
        mode="canonical_dedup",
        tta=True
    )

    predictions = engine.predict([
        SEQ_A,
        RC_A
    ])

    assert predictions[0] == predictions[1]

    stats = engine.stats()

    assert stats["model_evaluations"] == 1

    assert stats["encoder_evaluations"] == 2


def test_invalid_batch_does_not_modify_cache(
    invariant_model
):

    engine = make_engine(
        invariant_model,
        "canonical_dedup"
    )

    with pytest.raises(ValueError):

        engine.predict([
            SEQ_A,
            "INVALID"
        ])

    assert len(engine.cache) == 0

    assert engine.stats()["total_requests"] == 0