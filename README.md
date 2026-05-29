# LEGO Minifigures Image Classification & Error Diagnostics

## Overview
This repository contains the code and diagnostic tools for a LEGO minifigure image classification project (top 30 categories). It goes beyond traditional accuracy optimization by employing multimodal late fusion and Grad-CAM interpretability.

**Key Finding:** Misclassifications are largely driven by structural semantic ambiguity and label noise in the official LEGO commercial taxonomy, rather than computer vision failures.

**Key Results:**
- **Test Accuracy:** 0.891
- **Test Macro-F1:** 0.856

*(Completed as Assignment 2 for the Advanced Data Science and Deep Learning coursework.)*

## Repository Structure
```text
├── Phase 1 data exploration/      # Exploratory Data Analysis & overlap checks
├── Phase 2 splitting/             # Train/Val/Test (70/15/15) stratification
├── Phase 3 Top 30 trial/          # Base CV models (ConvNeXt, ResNet, ViT)
├── Phase 3 multimodel/            # Feature extraction (CLIP, DINOv2) & late fusion
├── Phase 3 ensemble/              # Final ensembling and Test-Time Augmentation (TTA)
├── Phase 4 Error diagnostics/     # Grad-CAM extraction & semantic ambiguity analysis
├── Phase 5 Presentation Assets/   # Visualizations & heatmaps
├── Assignment2_Final_Report.pdf   # Final academic report
└── README.md
```

## Methodology
1. **Base Deep Learning:** Fine-tuning `ConvNeXt-Tiny`, `ResNet-50`, and `ViT-B/16`.
2. **Multimodal Late Fusion:** Fusing embeddings from Foundation Models (`CLIP`, `DINOv2`) with metadata, classified via `XGBoost`.
3. **Ensemble & TTA:** Weighted ensemble combining CV logits and XGBoost probabilities, augmented with 8-crop TTA.

## How to Run
*(Requires placing the original `images/` and `.pt`/`.npz` files in the root directory)*

**Base model evaluation (e.g., ConvNeXt):**
```bash
python "Phase 3 Top 30 trial/top30_train_and_eval_v2.py"
```
**Final Ensemble with TTA:**
```bash
python "Phase 3 ensemble/ensemble_top30.py" --tta --tta-n 8
```
**Grad-CAM heatmaps for misclassified examples:**
```bash
python "Phase 4 Error diagnostics/extract_misclassified_attention.py"
```

## Note on Dataset Availability
The raw image dataset, large embedding files, and model weights are **not included** due to file size constraints and academic integrity policies. Full execution requires the original dataset.

## Author
**Zhaoyu Wang**
