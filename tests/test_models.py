import pytest
import torch

from src.models import (StandardCNN, InvariantCNN, build_model)

from src.sequence import reverse_complement_tensor

def random_dna(batch_size=8, length=500):
    indices = torch.randint(low=0, high=4, size=(batch_size, length))

    return torch.nn.functional.one_hot(indices, num_classes=4).permute(0, 2, 1).float()

@pytest.mark.parametrize(
    "model_name",
    ["standard", "augmented", "invariant"]
)
def test_output_shape(model_name):
    model = build_model(model_name)
    x = random_dna()
    output = model(x)
    assert output.shape == (8,)

def test_equal_parameter_counts():
    standard = StandardCNN()
    invariant = InvariantCNN()

    standard_params = sum(p.numel() for p in standard.parameters())
    invariant_params = sum(p.numel() for p in invariant.parameters())

    assert standard_params == invariant_params

@pytest.mark.parametrize(
    "seed",
    [42, 123, 456]
)
def test_prediction_invariance(seed):
    torch.manual_seed(seed)
    model = InvariantCNN()
    x = random_dna()
    rc = reverse_complement_tensor(x)
    
    with torch.inference_mode():
        original_predictions = model(x)
        rc_predictions = model(rc)
    
    torch.testing.assert_close(original_predictions, rc_predictions, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "model_name",
    ["standard", "augmented", "invariant"]
)
def test_gradient_flow(model_name):
    model = build_model(model_name)
    x = random_dna()
    labels = torch.randint(0, 2, (8,)).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(model(x), labels)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()