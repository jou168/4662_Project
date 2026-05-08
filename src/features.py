from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


TARGET_COLUMN = "AdoptionSpeed"
ID_COLUMNS = ("PetID",)
TEXT_COLUMNS = ("Name", "Description", "RescuerID")

BASE_CATEGORICAL_COLUMNS = (
    "Type",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "Health",
    "State",
)

BASE_NUMERIC_COLUMNS = (
    "Age",
    "Quantity",
    "Fee",
    "VideoAmt",
    "PhotoAmt",
)

ENGINEERED_NUMERIC_COLUMNS = (
    "description_length",
    "description_word_count",
    "has_description",
    "has_name",
    "is_mixed_breed",
    "color_count",
    "has_fee",
    "has_video",
    "has_photo",
)


def add_basic_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add low-leakage metadata features while leaving raw text for NLP models."""
    out = df.copy()

    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        if column in out:
            return out[column].fillna(default).astype(float)
        return pd.Series(default, index=out.index, dtype=float)

    description = out.get("Description", pd.Series("", index=out.index)).fillna("")
    name = out.get("Name", pd.Series("", index=out.index)).fillna("")

    out["description_length"] = description.str.len()
    out["description_word_count"] = description.str.split().map(len)
    out["has_description"] = description.str.strip().ne("").astype(np.int8)
    out["has_name"] = name.str.strip().ne("").astype(np.int8)

    out["is_mixed_breed"] = (numeric_series("Breed2") > 0).astype(np.int8)
    color_columns = [column for column in ("Color1", "Color2", "Color3") if column in out]
    out["color_count"] = (out[color_columns].fillna(0).astype(float) > 0).sum(axis=1)

    out["has_fee"] = (numeric_series("Fee") > 0).astype(np.int8)
    out["has_video"] = (numeric_series("VideoAmt") > 0).astype(np.int8)
    out["has_photo"] = (numeric_series("PhotoAmt") > 0).astype(np.int8)

    return out


def _existing_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


@dataclass
class TabularFeatureBuilder:
    """Leak-free transformer for tabular baseline features."""

    categorical_columns: tuple[str, ...] = BASE_CATEGORICAL_COLUMNS
    numeric_columns: tuple[str, ...] = BASE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

    def __post_init__(self) -> None:
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "TabularFeatureBuilder":
        features = add_basic_tabular_features(df)
        categorical = _existing_columns(features, self.categorical_columns)
        numeric = _existing_columns(features, self.numeric_columns)

        self.feature_names_ = categorical + numeric
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.preprocessor.fit(features, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("TabularFeatureBuilder must be fit before transform.")

        features = add_basic_tabular_features(df)
        matrix = self.preprocessor.transform(features).astype(np.float32)
        return pd.DataFrame(matrix, columns=self.feature_names_, index=df.index)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names_)


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_column].astype(int)
    x = df.drop(columns=[target_column])
    return x, y
