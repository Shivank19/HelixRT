import torch
import torch.nn as nn

from src.sequence import reverse_complement_tensor 

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=15,
                padding=7
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )

        self.output_dim = 64

    def forward(self, x):
        return self.layers(x)

class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()

        self.classifier = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

class InvariantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.classifier = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, x):
        rc = reverse_complement_tensor(x)
        combined = torch.cat([x, rc], dim=0)

        features = self.encoder(combined)

        forward_features, rc_features = (features.chunk(2, dim=0))

        invariant_features = (forward_features + rc_features) / 2.0

        logits = self.classifier(invariant_features)

        return logits.squeeze(-1)

def build_model(model_name):
    if model_name in ("standard", "augmented"):
        return StandardCNN()

    if model_name == "invariant":
        return InvariantCNN()

    raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    x = torch.randn(8, 4, 500)

    for name in ["standard", "augmented", "invariant"]:
        model = build_model(name)
        output = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel: {name}")
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)
        print("Trainable parameters:", params)