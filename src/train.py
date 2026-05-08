from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .features import TabularFeatureBuilder
from .utils import ensure_dir, quadratic_weighted_kappa


class MultimodalTensorDataset(Dataset):
    """Tensor dataset for precomputed tabular, image, and text features."""

    def __init__(
        self,
        tabular: np.ndarray,
        image: np.ndarray,
        text: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.tabular = torch.as_tensor(np.array(tabular, copy=True), dtype=torch.float32)
        self.image = torch.as_tensor(np.array(image, copy=True), dtype=torch.float32)
        self.text = torch.as_tensor(np.array(text, copy=True), dtype=torch.float32)
        self.labels = torch.as_tensor(np.array(labels, copy=True), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tabular[index], self.image[index], self.text[index], self.labels[index]


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)


class TorchClassifierTrainer:
    """Generic trainer for image-only or text-only torch classifiers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        labels: np.ndarray,
        learning_rate: float = 1e-3,
        checkpoint_dir: str | Path = "outputs/checkpoints",
        checkpoint_name: str | None = None,
        num_classes: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.checkpoint_name = checkpoint_name or f"{self.model.__class__.__name__}_best.pt"
        class_weights = compute_class_weights(labels, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        self.best_qwk = -np.inf

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.item())

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader: torch.utils.data.DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += float(loss.item())
                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                actuals.append(labels.cpu().numpy())

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(actuals)
        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=self.num_classes)
        return total_loss / max(len(val_loader), 1), qwk, y_pred, y_true

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        patience: int = 5,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_qwk, _, _ = self.validate(val_loader)
            self.scheduler.step(val_qwk)

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_qwk": val_qwk,
                }
            )

            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / self.checkpoint_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return history


class MultimodalTrainer:
    """Training loop for fusion models with QWK validation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        labels: np.ndarray,
        learning_rate: float = 1e-3,
        checkpoint_dir: str | Path = "outputs/checkpoints",
        checkpoint_name: str | None = None,
        num_classes: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.checkpoint_name = checkpoint_name or f"{self.model.__class__.__name__}_best.pt"
        class_weights = compute_class_weights(labels, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
        self.best_qwk = -np.inf

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for tabular, image, text, labels in train_loader:
            tabular = tabular.to(self.device)
            image = image.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(tabular, image, text)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader: torch.utils.data.DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []

        with torch.no_grad():
            for tabular, image, text, labels in val_loader:
                tabular = tabular.to(self.device)
                image = image.to(self.device)
                text = text.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(tabular, image, text)
                loss = self.criterion(outputs, labels)
                total_loss += float(loss.item())

                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                actuals.append(labels.cpu().numpy())

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(actuals)
        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=self.num_classes)
        return total_loss / max(len(val_loader), 1), qwk, y_pred, y_true

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        patience: int = 10,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_qwk, _, _ = self.validate(val_loader)
            self.scheduler.step(val_qwk)

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_qwk": val_qwk,
                }
            )

            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                patience_counter = 0
                checkpoint_path = self.checkpoint_dir / self.checkpoint_name
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return history


def cross_validate_tabular_classifier(
    model: object,
    x,
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Run leak-free tabular CV with the preprocessor fit inside each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    fold_predictions: list[np.ndarray] = []
    fold_actuals: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        feature_builder = TabularFeatureBuilder()
        x_train_features = feature_builder.fit_transform(x_train, y_train)
        x_val_features = feature_builder.transform(x_val)

        fold_model = clone(model)
        fold_model.fit(x_train_features, y_train)
        y_pred = fold_model.predict(x_val_features)
        qwk = quadratic_weighted_kappa(y_val.to_numpy(), y_pred)

        scores.append(qwk)
        fold_predictions.append(np.asarray(y_pred))
        fold_actuals.append(y_val.to_numpy())

        print(f"Fold {fold}/{n_splits} QWK: {qwk:.4f}")

    return {
        "fold_scores": scores,
        "mean_qwk": float(np.mean(scores)),
        "std_qwk": float(np.std(scores)),
        "predictions": fold_predictions,
        "actuals": fold_actuals,
    }


def cross_validate_text_classifier(
    model: object,
    texts: list[str],
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Run text-only CV for sklearn-style text classifiers."""
    labels = np.asarray(y)
    texts_array = np.asarray(texts, dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    fold_predictions: list[np.ndarray] = []
    fold_actuals: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_array, labels), start=1):
        fold_model = clone(model)
        fold_model.fit(texts_array[train_idx].tolist(), labels[train_idx])
        y_pred = fold_model.predict(texts_array[val_idx].tolist())
        y_val = labels[val_idx]
        qwk = quadratic_weighted_kappa(y_val, y_pred)

        scores.append(qwk)
        fold_predictions.append(np.asarray(y_pred))
        fold_actuals.append(y_val)

        print(f"Fold {fold}/{n_splits} QWK: {qwk:.4f}")

    return {
        "fold_scores": scores,
        "mean_qwk": float(np.mean(scores)),
        "std_qwk": float(np.std(scores)),
        "predictions": fold_predictions,
        "actuals": fold_actuals,
    }
