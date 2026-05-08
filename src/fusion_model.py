from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusionModel(nn.Module):
    """Concatenate all modality embeddings before a shared classifier."""

    def __init__(
        self,
        tabular_dim: int,
        image_dim: int,
        text_dim: int,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        fused_dim = tabular_dim + image_dim + text_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([tabular, image, text], dim=1))


class LateFusionModel(nn.Module):
    """Project each modality separately before fusion."""

    def __init__(
        self,
        tabular_dim: int,
        image_dim: int,
        text_dim: int,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.tabular_net = nn.Sequential(nn.Linear(tabular_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64))
        self.image_net = nn.Sequential(nn.Linear(image_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128))
        self.text_net = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128))
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        tabular_features = self.tabular_net(tabular)
        image_features = self.image_net(image)
        text_features = self.text_net(text)
        return self.classifier(torch.cat([tabular_features, image_features, text_features], dim=1))


class IntermediateFusionModel(nn.Module):
    """Project modalities to a shared space and fuse them with self-attention."""

    def __init__(
        self,
        tabular_dim: int,
        image_dim: int,
        text_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.tabular_projection = nn.Linear(tabular_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        modality_stack = torch.stack(
            [
                self.tabular_projection(tabular),
                self.image_projection(image),
                self.text_projection(text),
            ],
            dim=1,
        )
        attended, _ = self.attention(modality_stack, modality_stack, modality_stack)
        return self.classifier(attended.reshape(attended.size(0), -1))


def build_fusion_model(
    architecture: str,
    tabular_dim: int,
    image_dim: int,
    text_dim: int,
    num_classes: int = 5,
) -> nn.Module:
    """Create a fusion model by config/notebook architecture name."""
    normalized = architecture.lower().replace("-", "_")
    if normalized == "early":
        return EarlyFusionModel(tabular_dim, image_dim, text_dim, num_classes=num_classes)
    if normalized == "late":
        return LateFusionModel(tabular_dim, image_dim, text_dim, num_classes=num_classes)
    if normalized in {"intermediate", "intermediate_attention", "attention"}:
        return IntermediateFusionModel(tabular_dim, image_dim, text_dim, num_classes=num_classes)
    raise ValueError(f"Unknown fusion architecture: {architecture}")
