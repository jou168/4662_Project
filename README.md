# PetFinder.my Adoption Prediction

Multimodal adoption-speed prediction for the Kaggle PetFinder.my dataset.

## Project Summary

This repository contains the completed CS4662 PetFinder.my adoption-speed project. The
pipeline predicts the ordinal `AdoptionSpeed` target from structured profile metadata,
free-text descriptions, and pet images using modular source code in `src/` and report
notebooks in `notebooks/`.

The final report is `notebooks/final_report.ipynb`. Report tables are generated from
CSV artifacts in `outputs/reports/`, and multimodal fusion uses cached feature matrices
in `outputs/features/`.

## Final Results

- XGBoost tabular 5-fold CV QWK: `0.3658 +/- 0.0242`.
- LightGBM tabular 5-fold CV QWK: `0.3568 +/- 0.0206`.
- TF-IDF + LinearSVC text 5-fold CV QWK: `0.2383 +/- 0.0219`.
- ResNet50 image pilot best validation QWK: `0.1810`.
- Deep early fusion with tabular + BERT + ResNet50 embeddings best validation QWK:
  `0.3285`.
- Deep ablation deltas vs. full fusion: no tabular `-0.0407`, no image `-0.0618`,
  no text `-0.0416`.

Cached feature matrices used by fusion:

- BERT/multilingual transformer description embeddings: `(14993, 768)`.
- ResNet50 image embeddings: `(14993, 2048)`.

Generated reports, checkpoints, and cached embeddings are stored under `outputs/`,
which is intentionally ignored except for `outputs/.gitkeep`.

## Repository Layout

```text
src/
  data.py          # Path handling, CSV loading, splits, modality inventory
  features.py      # Leak-free tabular feature engineering/preprocessing
  image_model.py   # Image dataset and ResNet50 embedding extraction
  text_model.py    # Transformer description embedding extraction
  fusion_model.py  # Early, late, and attention-based fusion models
  train.py         # Multimodal training loop
  utils.py         # Metrics, seeding, filesystem helpers
notebooks/         # Readable experiment/final-report notebooks
configs/           # Experiment configuration
outputs/           # Checkpoints, cached embeddings, reports, predictions
Data/              # Local Kaggle data, ignored for future commits
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```
