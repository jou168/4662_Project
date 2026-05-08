from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def descriptions_from_frame(
    df: pd.DataFrame,
    text_column: str = "Description",
    fallback: str = "",
) -> list[str]:
    return df.get(text_column, pd.Series(fallback, index=df.index)).fillna(fallback).astype(str).tolist()


def build_tfidf_svm_classifier(
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Build the text baseline: TF-IDF plus linear SVM."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            ("classifier", LinearSVC(class_weight=class_weight, random_state=42)),
        ]
    )


@dataclass(frozen=True)
class TransformerTextConfig:
    model_name: str = "bert-base-multilingual-cased"
    max_length: int = 160
    batch_size: int = 16


class TransformerTextEmbedder:
    """Extract CLS-token embeddings for pet descriptions."""

    def __init__(
        self,
        config: TransformerTextConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformerTextEmbedder requires the optional 'transformers' package. "
                "Install project dependencies from requirements.txt before extracting text embeddings."
            ) from exc

        self.config = config or TransformerTextConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                batches.append(cls_embeddings)

        return np.vstack(batches).astype(np.float32)
