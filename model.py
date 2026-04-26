import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# CNN version for CIFAR10 image classification.
class CIFAR10CNN(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        block_dropout = min(dropout, 0.2)

        # Lightweight ResNet-style backbone.
        # Input [B, 3, 32, 32] -> output [B, 384, 4, 4].
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48, 48, stride=1, dropout=block_dropout),
            ResidualBlock(48, 48, stride=1, dropout=block_dropout),
            ResidualBlock(48, 96, stride=2, dropout=block_dropout),
            ResidualBlock(96, 96, stride=1, dropout=block_dropout),
            ResidualBlock(96, 192, stride=2, dropout=block_dropout),
            ResidualBlock(192, 192, stride=1, dropout=block_dropout),
            ResidualBlock(192, 384, stride=2, dropout=block_dropout),
            ResidualBlock(384, 384, stride=1, dropout=block_dropout),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name: str = "cnn", dropout: float = 0.3) -> nn.Module:
    if model_name.lower() != "cnn":
        raise ValueError(f"Unknown model: {model_name}. This project now supports only: cnn")
    return CIFAR10CNN(dropout=dropout)


# Backward-compatible aliases for old imports.
MNISTCNN = CIFAR10CNN
MNISTNet = CIFAR10CNN
