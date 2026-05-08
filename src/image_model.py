from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

def build_pet_image_index(image_dir: str | Path) -> dict[str, list[Path]]:
    """Index all pet images in one directory scan."""
    index: dict[str, list[Path]] = {}
    for image_path in Path(image_dir).glob("*.jpg"):
        pet_id = image_path.stem.rsplit("-", 1)[0]
        index.setdefault(pet_id, []).append(image_path)

    for image_paths in index.values():
        image_paths.sort()
    return index


class PetImageDataset(Dataset):
    """Load one representative image per pet, with a blank fallback."""

    def __init__(
        self,
        pet_ids: list[str] | np.ndarray | pd.Series,
        image_dir: str | Path,
        transform: transforms.Compose | None = None,
        image_size: int = 224,
    ) -> None:
        self.pet_ids = [str(pet_id) for pet_id in pet_ids]
        self.image_dir = Path(image_dir)
        self.transform = transform or build_image_transform(image_size=image_size, train=False)
        self.image_size = image_size
        self.image_index = build_pet_image_index(self.image_dir)

    def __len__(self) -> int:
        return len(self.pet_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        pet_id = self.pet_ids[index]
        image_paths = self.image_index.get(pet_id, [])

        if image_paths:
            image = Image.open(image_paths[0]).convert("RGB")
        else:
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))

        return self.transform(image), pet_id


class PetImageClassificationDataset(Dataset):
    """Load one representative image per labeled pet for image-only training."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        pet_id_column: str = "PetID",
        target_column: str = "AdoptionSpeed",
        transform: transforms.Compose | None = None,
        image_size: int = 224,
    ) -> None:
        self.frame = frame[[pet_id_column, target_column]].reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.pet_id_column = pet_id_column
        self.target_column = target_column
        self.transform = transform or build_image_transform(image_size=image_size, train=False)
        self.image_size = image_size
        self.image_index = build_pet_image_index(self.image_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        pet_id = str(row[self.pet_id_column])
        image_paths = self.image_index.get(pet_id, [])

        if image_paths:
            image = Image.open(image_paths[0]).convert("RGB")
        else:
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))

        label = torch.tensor(int(row[self.target_column]), dtype=torch.long)
        return self.transform(image), label


def build_image_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    steps: list[object] = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


def build_resnet50_feature_extractor(
    device: torch.device | str | None = None,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    if device is not None:
        feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def build_resnet50_classifier(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build an image-only ResNet50 classifier for adoption speed."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def extract_image_embeddings(
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> pd.DataFrame:
    """Extract one embedding row per pet from a feature extractor."""
    feature_extractor.eval()
    embeddings: list[np.ndarray] = []
    pet_ids: list[str] = []

    with torch.no_grad():
        for images, batch_pet_ids in loader:
            images = images.to(device)
            features = feature_extractor(images).view(images.size(0), -1)
            embeddings.append(features.cpu().numpy())
            pet_ids.extend(batch_pet_ids)

    matrix = np.vstack(embeddings).astype(np.float32)
    columns = [f"image_{index}" for index in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=pet_ids, columns=columns)


def image_presence_summary(
    pet_ids: list[str] | np.ndarray | pd.Series,
    image_dir: str | Path,
) -> dict[str, int | float]:
    """Summarize how many pets have at least one image."""
    ids = [str(pet_id) for pet_id in pet_ids]
    image_index = build_pet_image_index(image_dir)
    counts = [len(image_index.get(pet_id, [])) for pet_id in ids]
    has_image = [count > 0 for count in counts]
    return {
        "pets": len(ids),
        "pets_with_images": int(np.sum(has_image)),
        "pets_without_images": int(len(ids) - np.sum(has_image)),
        "total_images": int(np.sum(counts)),
        "mean_images_per_pet": float(np.mean(counts)) if counts else 0.0,
    }
