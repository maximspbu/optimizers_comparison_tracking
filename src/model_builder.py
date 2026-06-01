from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .config import DEVICE


# ---------------------------------------------------------------------------
# Generic parametric MLP (legacy backbone)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Parametric feed-forward network for tabular tasks.

    ``depth`` counts hidden layers: depth=2 gives
    ``input → hidden → hidden → output``.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        depth: int = 2,
        activation_function: Optional[nn.Module] = nn.ReLU(),
        normalization: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        layers: list[nn.Module] = [nn.Linear(input_shape, hidden_units, device=device)]
        for _ in range(depth - 1):
            if normalization is not None:
                layers.append(normalization)
            if activation_function is not None:
                layers.append(activation_function)
            if dropout is not None:
                layers.append(dropout)
            layers.append(nn.Linear(hidden_units, hidden_units, device=device))
        if activation_function is not None:
            layers.append(activation_function)
        layers.append(nn.Linear(hidden_units, output_shape, device=device))
        self.block = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.block(X)


class SimpleRegressionModel(MLP):
    """Backwards-compatible 2-layer regression MLP."""

    def __init__(
        self,
        input_shape: int = 8,
        hidden_units: int = 64,
        output_shape: int = 1,
        activation_function: Optional[nn.Module] = nn.ReLU(),
        normalization: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            hidden_units=hidden_units,
            output_shape=output_shape,
            depth=2,
            activation_function=activation_function,
            normalization=normalization,
            dropout=dropout,
            device=device,
        )


class SimpleClassificationModel(MLP):
    """Backwards-compatible 1-layer classification model."""

    def __init__(
        self,
        input_shape: int = 8,
        hidden_units: int = 16,
        output_shape: int = 2,
        activation_function: Optional[nn.Module] = None,
        normalization: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            hidden_units=hidden_units,
            output_shape=output_shape,
            depth=1,
            activation_function=activation_function,
            normalization=normalization,
            dropout=dropout,
            device=device,
        )


class DeepClassificationModel(MLP):
    """Backwards-compatible 8-layer classification MLP."""

    def __init__(
        self,
        input_shape: int = 8,
        hidden_units: int = 16,
        output_shape: int = 2,
        activation_function: Optional[nn.Module] = None,
        normalization: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            hidden_units=hidden_units,
            output_shape=output_shape,
            depth=8,
            activation_function=activation_function,
            normalization=normalization,
            dropout=dropout,
            device=device,
        )


MLP_DEPTHS: dict[str, int] = {
    "mlp_shallow": 2,
    "mlp_medium": 4,
    "mlp_deep": 8,
}


# ---------------------------------------------------------------------------
# SimpleMLP for regression (notebook-style, BN + GELU + Dropout)
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Feed-forward MLP for tabular regression (regression_work.ipynb style).

    Architecture: [Linear → BN → GELU → Dropout] × len(hidden_dims) → Linear(→1).
    Output is squeezed to 1-D so loss = MSELoss(preds, y) with scalar y works.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """Residual MLP for large tabular regression datasets."""

    def __init__(
        self,
        in_features: int,
        width: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# AttentionMLP for regression (feature-wise Transformer)
# ---------------------------------------------------------------------------


class AttentionMLP(nn.Module):
    """Feature-wise Transformer for tabular regression.

    Each scalar input feature is embedded to d_model, treated as a token.
    Token representations are mean-pooled then passed through a regression head.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Ensure divisibility
        d_model = max(d_model, nhead)
        d_model = (d_model // nhead) * nhead

        self.feature_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)          # (B, F) → (B, F, 1)
        x = self.feature_proj(x)      # (B, F, d_model)
        x = self.transformer(x)       # (B, F, d_model)
        x = x.mean(dim=1)             # mean-pool → (B, d_model)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# TabularAttentionModel for classification (FT-Transformer with CLS token)
# ---------------------------------------------------------------------------


class TabularAttentionModel(nn.Module):
    """FT-Transformer-style model for tabular classification (classification-1-10.ipynb).

    Each input feature is linearly projected to a token embedding.
    A CLS token aggregates information via multi-head self-attention.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int = 64,
        output_shape: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_units = max(hidden_units, num_heads)
        hidden_units = (hidden_units // num_heads) * num_heads

        self.feature_proj = nn.Linear(1, hidden_units)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_units))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_units),
            nn.Linear(hidden_units, output_shape),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.feature_proj(x.unsqueeze(-1))    # (B, D, hidden)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, hidden)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, D+1, hidden)
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0])                   # CLS → logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TABULAR_BUILDERS = {
    **{k: k for k in MLP_DEPTHS},  # mlp_shallow, mlp_medium, mlp_deep
    "simple_mlp": "simple_mlp",
    "residual_mlp": "residual_mlp",
    "attention_mlp": "attention_mlp",
    "simple_cls": "simple_cls",
    "attention_cls": "attention_cls",
}


def build_tabular_model(
    name: str,
    input_shape: int,
    output_shape: int,
    hidden_units: int = 64,
    **kwargs,
) -> nn.Module:
    """Factory for tabular models keyed by name."""
    if name in MLP_DEPTHS:
        return MLP(
            input_shape=input_shape,
            hidden_units=hidden_units,
            output_shape=output_shape,
            depth=MLP_DEPTHS[name],
            **kwargs,
        )
    if name == "simple_mlp":
        return SimpleMLP(in_features=input_shape)
    if name == "residual_mlp":
        return ResidualMLP(in_features=input_shape)
    if name == "attention_mlp":
        return AttentionMLP(in_features=input_shape)
    if name == "simple_cls":
        return SimpleClassificationModel(
            input_shape=input_shape,
            hidden_units=max(64, input_shape),
            output_shape=output_shape,
            activation_function=nn.ReLU(),
        )
    if name == "attention_cls":
        return TabularAttentionModel(
            input_shape=input_shape,
            hidden_units=max(64, input_shape),
            output_shape=output_shape,
        )
    raise ValueError(f"unknown tabular model {name!r}; choose from {list(_TABULAR_BUILDERS)}")


def list_tabular_models() -> tuple[str, ...]:
    return tuple(_TABULAR_BUILDERS)
