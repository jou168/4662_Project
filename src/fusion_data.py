from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .data import ProjectPaths, list_pet_images
from .features import TabularFeatureBuilder, split_features_target
from .text_model import descriptions_from_frame


@dataclass(frozen=True)
class FusionFeatureSet:
    """Aligned multimodal arrays keyed by PetID."""

    pet_ids: np.ndarray
    tabular: np.ndarray
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray

    @property
    def tabular_dim(self) -> int:
        return int(self.tabular.shape[1])

    @property
    def image_dim(self) -> int:
        return int(self.image.shape[1])

    @property
    def text_dim(self) -> int:
        return int(self.text.shape[1])

    def without(self, *modalities: str) -> "FusionFeatureSet":
        """Return a copy with selected modalities zeroed for ablation."""
        blocked = set(modalities)
        return FusionFeatureSet(
            pet_ids=self.pet_ids.copy(),
            tabular=np.zeros_like(self.tabular) if "tabular" in blocked else self.tabular.copy(),
            image=np.zeros_like(self.image) if "image" in blocked else self.image.copy(),
            text=np.zeros_like(self.text) if "text" in blocked else self.text.copy(),
            labels=self.labels.copy(),
        )


def load_npz_embeddings(
    path: str | Path,
    pet_ids: pd.Series | np.ndarray | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load `.npz` embeddings with `pet_ids` and align them to requested IDs."""
    data = np.load(path, allow_pickle=True)
    source_ids = data["pet_ids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    requested_ids = np.asarray(pet_ids).astype(str)

    id_to_index = {pet_id: index for index, pet_id in enumerate(source_ids)}
    aligned = np.zeros((len(requested_ids), embeddings.shape[1]), dtype=np.float32)
    present = np.zeros(len(requested_ids), dtype=bool)

    for row_index, pet_id in enumerate(requested_ids):
        source_index = id_to_index.get(pet_id)
        if source_index is not None:
            aligned[row_index] = embeddings[source_index]
            present[row_index] = True

    return aligned, present


def image_metadata_frame(
    pet_ids: pd.Series | np.ndarray | list[str],
    image_dir: str | Path,
) -> pd.DataFrame:
    """Build lightweight image-availability features from local image files."""
    rows: list[dict[str, float]] = []
    index = [str(pet_id) for pet_id in pet_ids]

    for pet_id in index:
        image_paths = list_pet_images(pet_id, image_dir)
        sizes = [path.stat().st_size for path in image_paths if path.exists()]
        image_count = len(image_paths)
        total_bytes = float(np.sum(sizes)) if sizes else 0.0
        rows.append(
            {
                "image_count": float(image_count),
                "has_image": float(image_count > 0),
                "total_image_bytes": total_bytes,
                "mean_image_bytes": total_bytes / image_count if image_count else 0.0,
            }
        )

    return pd.DataFrame(rows, index=index)


def fit_transform_image_metadata(
    train_pet_ids: pd.Series | np.ndarray | list[str],
    val_pet_ids: pd.Series | np.ndarray | list[str],
    image_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    train_frame = image_metadata_frame(train_pet_ids, image_dir)
    val_frame = image_metadata_frame(val_pet_ids, image_dir)
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_frame).astype(np.float32),
        scaler.transform(val_frame).astype(np.float32),
    )


def fit_transform_tfidf_svd_text(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int = 64,
    max_features: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create dense text embeddings from TF-IDF followed by SVD."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    train_texts = descriptions_from_frame(train_df)
    val_texts = descriptions_from_frame(val_df)
    train_tfidf = vectorizer.fit_transform(train_texts)
    val_tfidf = vectorizer.transform(val_texts)

    feature_count = train_tfidf.shape[1]
    if feature_count <= 1:
        train_dense = train_tfidf.toarray().astype(np.float32)
        val_dense = val_tfidf.toarray().astype(np.float32)
    else:
        effective_components = min(n_components, feature_count - 1)
        svd = TruncatedSVD(n_components=effective_components, random_state=42)
        train_dense = svd.fit_transform(train_tfidf).astype(np.float32)
        val_dense = svd.transform(val_tfidf).astype(np.float32)

    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_dense).astype(np.float32),
        scaler.transform(val_dense).astype(np.float32),
    )


def build_lightweight_fusion_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    paths: ProjectPaths,
    text_components: int = 64,
    max_text_features: int = 50000,
) -> tuple[FusionFeatureSet, FusionFeatureSet]:
    """Build a runnable real-data fusion baseline without heavy deep embeddings."""
    train_x, train_y = split_features_target(train_df)
    val_x, val_y = split_features_target(val_df)

    tabular_builder = TabularFeatureBuilder()
    train_tabular = tabular_builder.fit_transform(train_x, train_y).to_numpy(dtype=np.float32)
    val_tabular = tabular_builder.transform(val_x).to_numpy(dtype=np.float32)

    train_image, val_image = fit_transform_image_metadata(
        train_df["PetID"],
        val_df["PetID"],
        paths.train_images_dir,
    )
    train_text, val_text = fit_transform_tfidf_svd_text(
        train_df,
        val_df,
        n_components=text_components,
        max_features=max_text_features,
    )

    train_features = FusionFeatureSet(
        pet_ids=train_df["PetID"].astype(str).to_numpy(),
        tabular=train_tabular,
        image=train_image,
        text=train_text,
        labels=train_y.to_numpy(dtype=np.int64),
    )
    val_features = FusionFeatureSet(
        pet_ids=val_df["PetID"].astype(str).to_numpy(),
        tabular=val_tabular,
        image=val_image,
        text=val_text,
        labels=val_y.to_numpy(dtype=np.int64),
    )
    return train_features, val_features


def build_cached_embedding_fusion_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_embedding_path: str | Path,
    image_embedding_path: str | Path,
) -> tuple[FusionFeatureSet, FusionFeatureSet, dict[str, int]]:
    """Build fusion features from cached BERT/ResNet embeddings aligned by PetID."""
    train_x, train_y = split_features_target(train_df)
    val_x, val_y = split_features_target(val_df)

    tabular_builder = TabularFeatureBuilder()
    train_tabular = tabular_builder.fit_transform(train_x, train_y).to_numpy(dtype=np.float32)
    val_tabular = tabular_builder.transform(val_x).to_numpy(dtype=np.float32)

    train_text, train_text_present = load_npz_embeddings(text_embedding_path, train_df["PetID"])
    val_text, val_text_present = load_npz_embeddings(text_embedding_path, val_df["PetID"])
    train_image, train_image_present = load_npz_embeddings(image_embedding_path, train_df["PetID"])
    val_image, val_image_present = load_npz_embeddings(image_embedding_path, val_df["PetID"])

    coverage = {
        "train_text_present": int(train_text_present.sum()),
        "val_text_present": int(val_text_present.sum()),
        "train_image_present": int(train_image_present.sum()),
        "val_image_present": int(val_image_present.sum()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }

    train_features = FusionFeatureSet(
        pet_ids=train_df["PetID"].astype(str).to_numpy(),
        tabular=train_tabular,
        image=train_image.astype(np.float32),
        text=train_text.astype(np.float32),
        labels=train_y.to_numpy(dtype=np.int64),
    )
    val_features = FusionFeatureSet(
        pet_ids=val_df["PetID"].astype(str).to_numpy(),
        tabular=val_tabular,
        image=val_image.astype(np.float32),
        text=val_text.astype(np.float32),
        labels=val_y.to_numpy(dtype=np.int64),
    )
    return train_features, val_features, coverage
