"""Factory for vision architectures used in image-classification experiments.

Three architectures: ``resnet18`` (medium depth ~11M params), ``small_cnn``
(lightweight ~100k params), ``efficientnet_b0`` (~5M params).

All models accept ``in_channels`` so FashionMNIST (1 channel) works alongside
CIFAR-10/100 (3 channels).
"""

from typing import Callable

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Lightweight 3-block CNN with GAP head. Around 100k parameters."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _adapt_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        bias=conv.bias is not None,
    )


def _build_resnet18(num_classes: int, in_channels: int) -> nn.Module:
    from torchvision.models import resnet18

    model = resnet18(num_classes=num_classes)
    model.conv1 = _adapt_in_channels(model.conv1, in_channels)
    return model


def _build_efficientnet_b0(num_classes: int, in_channels: int) -> nn.Module:
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(num_classes=num_classes)
    first_conv = model.features[0][0]
    if isinstance(first_conv, nn.Conv2d) and first_conv.in_channels != in_channels:
        model.features[0][0] = _adapt_in_channels(first_conv, in_channels)
    return model


def _build_small_cnn(num_classes: int, in_channels: int) -> nn.Module:
    return SmallCNN(num_classes=num_classes, in_channels=in_channels)


def _build_efficientnet_v2_s(num_classes: int, in_channels: int) -> nn.Module:
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # Adapt first conv for non-3-channel input
    first_conv = model.features[0][0]
    if isinstance(first_conv, nn.Conv2d) and first_conv.in_channels != in_channels:
        model.features[0][0] = _adapt_in_channels(first_conv, in_channels)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


_BUILDERS: dict[str, Callable[[int, int], nn.Module]] = {
    "resnet18": _build_resnet18,
    "small_cnn": _build_small_cnn,
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet_v2_s": _build_efficientnet_v2_s,
}


def build_vision_model(name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"unknown vision model {name!r}; choose from {list(_BUILDERS)}")
    return _BUILDERS[name](num_classes, in_channels)


def list_vision_models() -> tuple[str, ...]:
    return tuple(_BUILDERS)
