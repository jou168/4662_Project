from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "AdoptionSpeed"
PET_ID_COLUMN = "PetID"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for the local project layout."""

    root: Path
    data_dir: Path
    outputs_dir: Path

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train" / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test" / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.data_dir / "test" / "sample_submission.csv"

    @property
    def train_images_dir(self) -> Path:
        return self.data_dir / "train_images"

    @property
    def test_images_dir(self) -> Path:
        return self.data_dir / "test_images"

    @property
    def train_metadata_dir(self) -> Path:
        return self.data_dir / "train_metadata"

    @property
    def test_metadata_dir(self) -> Path:
        return self.data_dir / "test_metadata"

    @property
    def train_sentiment_dir(self) -> Path:
        return self.data_dir / "train_sentiment"

    @property
    def test_sentiment_dir(self) -> Path:
        return self.data_dir / "test_sentiment"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_paths(
    root: str | Path | None = None,
    data_dir: str | Path = "Data",
    outputs_dir: str | Path = "outputs",
) -> ProjectPaths:
    root_path = Path(root).resolve() if root is not None else project_root()
    return ProjectPaths(
        root=root_path,
        data_dir=(root_path / data_dir).resolve(),
        outputs_dir=(root_path / outputs_dir).resolve(),
    )


def load_train_data(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or get_paths()
    return pd.read_csv(paths.train_csv)


def load_test_data(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or get_paths()
    return pd.read_csv(paths.test_csv)


def load_label_tables(paths: ProjectPaths | None = None) -> dict[str, pd.DataFrame]:
    """Load canonical breed, color, and state lookup tables when present."""
    paths = paths or get_paths()
    candidates = {
        "breed": ["breed_labels.csv", "BreedLabels.csv", "PetFinder-BreedLabels.csv"],
        "color": ["color_labels.csv", "ColorLabels.csv", "PetFinder-ColorLabels.csv"],
        "state": ["state_labels.csv", "StateLabels.csv", "PetFinder-StateLabels.csv"],
    }
    tables: dict[str, pd.DataFrame] = {}
    for table_name, filenames in candidates.items():
        for filename in filenames:
            path = paths.data_dir / filename
            if path.exists():
                tables[table_name] = pd.read_csv(path)
                break
    return tables


def make_stratified_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labeled dataframe while preserving the adoption-speed distribution."""
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def list_pet_images(pet_id: str, image_dir: str | Path) -> list[Path]:
    return sorted(Path(image_dir).glob(f"{pet_id}-*.jpg"))


def load_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_sentiment(pet_id: str, sentiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(sentiment_dir) / f"{pet_id}.json"
    if not path.exists():
        return None
    return load_json_file(path)


def modality_counts(paths: ProjectPaths | None = None) -> dict[str, int]:
    """Return a quick inventory of the local Kaggle files."""
    paths = paths or get_paths()
    directories = {
        "train_images": paths.train_images_dir,
        "test_images": paths.test_images_dir,
        "train_metadata": paths.train_metadata_dir,
        "test_metadata": paths.test_metadata_dir,
        "train_sentiment": paths.train_sentiment_dir,
        "test_sentiment": paths.test_sentiment_dir,
    }
    return {
        name: sum(1 for _ in directory.glob("*")) if directory.exists() else 0
        for name, directory in directories.items()
    }

