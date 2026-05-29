# LEGO Minifigures Image Classification & Error Diagnostics

## Overview

This project develops a deep learning pipeline to classify LEGO minifigures into the 30 most frequent categories. Beyond optimizing accuracy, it investigates **why** certain classifications fail through multimodal analysis and Grad-CAM diagnostics.

**Key Insight:** Most misclassifications stem from semantic ambiguity and label noise in LEGO's official taxonomy—not computer vision limitations.

### Results
- **Test Accuracy:** 89.1%
- **Test Macro-F1:** 85.6%

## Repository Structure

```
├── Phase 1 data exploration/      # EDA and category overlap analysis
├── Phase 2 splitting/             # Train/Val/Test stratification (70/15/15)
├── Phase 3 Top 30 trial/          # Base models: ConvNeXt, ResNet, ViT
├── Phase 3 multimodel/            # Multimodal embeddings (CLIP, DINOv2) + XGBoost
├── Phase 3 ensemble/              # Weighted ensemble + Test-Time Augmentation
├── Phase 4 Error diagnostics/     # Grad-CAM & semantic ambiguity analysis
├── Phase 5 Presentation Assets/   # Visualizations and confusion matrices
├── Assignment2_Final_Report.pdf   # Complete academic report
└── README.md
```

## Methodology

1. **Base Models:** Fine-tune ConvNeXt-Tiny, ResNet-50, ViT-B/16
2. **Multimodal Fusion:** Extract embeddings from foundation models (CLIP, DINOv2), combine with metadata, classify via XGBoost
3. **Ensemble & TTA:** Weighted ensemble + 8-crop test-time augmentation for robust predictions

## Quick Start

**Note:** Requires the original image dataset, metadata files, and embeddings placed in the project root.

```bash
# Train/evaluate a base model
python "Phase 3 Top 30 trial/top30_train_and_eval_v2.py"

# Run final ensemble with TTA
python "Phase 3 ensemble/ensemble_top30.py" --tta --tta-n 8

# Generate Grad-CAM visualizations
python "Phase 4 Error diagnostics/extract_misclassified_attention.py"
```

## Dataset

The dataset comprises web-scraped LEGO minifigure images with metadata (identifiers, release year, commercial category).

**Challenges:**
- Visually similar minifigures across categories
- Overlapping NINJAGO-related labels
- Broad categories (Town, Holiday & Event)
- Potential label noise in LEGO's taxonomy

**Availability:** Raw images, embedding files (.npz), and model weights (.pt) are **not included** due to file size constraints and academic integrity policies. Full reproduction requires the original dataset or equivalent reconstruction.

## Key Findings

| Category Type | Performance | Notes |
|---|---|---|
| Visually Distinctive | High recall | Minecraft, DUPLO, Fabuland, NEXO KNIGHTS |
| Visually Similar | Lower recall | NINJAGO variants, LEGO Brand, Holiday & Event |

Grad-CAM analysis shows models focus on meaningful regions (head, torso, accessories) rather than background noise.

## Dependencies

```
Python 3.10+
PyTorch & torchvision
scikit-learn, XGBoost
NumPy, pandas, Pillow, matplotlib
```

## Author

**Zhaoyu Wang**  
Advanced Data Science and Deep Learning (Assignment 2)
