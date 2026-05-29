# Deep Learning on Images: Classification and Error Diagnostics of LEGO Minifigures

## Overview

This repository contains the code, analysis, and diagnostic tools for a LEGO minifigure image classification project. The task is to classify web-scraped LEGO minifigure images into the 30 most frequent categories.

The project does not only focus on improving predictive performance. It also investigates why certain classes are difficult to classify. In particular, the diagnostic analysis suggests that many remaining errors are attributable to overlapping commercial categories, visually similar minifigure designs, and potential label noise in the original dataset.


## Repository Structure

The repository is organized chronologically to reflect the main research pipeline:

```text
├── Phase 1 data exploration/      # Exploratory data analysis and category overlap checks
├── Phase 2 splitting/             # Train/validation/test stratification scripts
├── Phase 3 Top 30 trial/          # Training and evaluation scripts for base vision models
├── Phase 3 multimodel/            # Feature extraction and multimodal late-fusion models
├── Phase 3 ensemble/              # Final model ensembling and test-time augmentation
├── Phase 4 Error diagnostics/     # Grad-CAM, misclassification analysis, and semantic diagnostics
├── Phase 5 Presentation Assets/   # Final visualizations, heatmaps, and confusion matrices
└── README.md                      # Project documentation
```

## Dataset

The dataset consists of web-scraped LEGO minifigure images and accompanying metadata. The metadata includes information such as minifigure identifiers, release year, estimated value, and commercial categories.

The classification task is restricted to the top 30 most frequent categories in order to reduce the extreme long-tail structure of the original dataset while keeping the task sufficiently challenging.

The dataset contains several sources of classification difficulty:

- visually similar minifigures across different categories;
- overlapping franchise-related labels, such as NINJAGO-related categories;
- broad commercial categories such as Town or Holiday & Event;
- potential label noise caused by LEGO's official commercial taxonomy rather than purely visual differences.

### Dataset Availability

The raw image dataset (images/), large embedding files (.npz), and trained model weights (.pt) are not included in this GitHub repository due to file size constraints.

As a result, cloning this repository alone is not sufficient to reproduce the full pipeline from scratch. Full execution requires access to the original dataset or a reconstructed version following the original web-scraping methodology.

However, the repository includes the source code, methodology, model configuration, evaluation scripts, diagnostic tools, and reported results. Therefore, the workflow and analytical procedure remain fully transparent and reproducible in principle.

## Methodology

The project follows an end-to-end image classification pipeline:

### Data Exploration
The metadata and category distributions were analyzed to identify long-tail class imbalance, overlapping categories, and potential sources of ambiguity.

### Dataset Splitting
A stratified 70/15/15 train-validation-test split was used to preserve class distributions across the three subsets.

### Base Deep Learning Models
Several pretrained computer vision architectures were fine-tuned for the 30-class classification task.

### Multimodal Late Fusion
Visual embeddings from pretrained foundation models were combined with selected non-leaky metadata features. XGBoost classifiers were then trained on these fused representations.

### Weighted Ensemble
Predictions from multiple models were combined using validation-tuned weighted averaging.

### Error Diagnostics
Confusion matrices, metadata cross-checks, and Grad-CAM visualizations were used to investigate the structure of remaining errors.

## Models

The project evaluates several model families:

### Base Computer Vision Models
- ResNet-50
- ViT-B/16
- ConvNeXt-Tiny

These models were fine-tuned using image transformations, validation monitoring, and macro-F1-based evaluation.

### Embedding-Based Models

Visual embeddings were extracted using pretrained representation models, including:

- CLIP
- DINOv2
- ResNet-based embeddings
- ViT-based embeddings

These embeddings were combined with selected metadata features and classified using XGBoost.

### Ensemble Model

The final ensemble combines predictions from several base and embedding-based models using validation-optimized weights. The ensemble is designed to reduce dependence on a single model architecture and leverage complementary prediction strengths.

## Evaluation

The main evaluation metrics are:

- accuracy;
- macro-F1 score;
- per-class precision, recall, and F1-score;
- confusion matrices;
- Grad-CAM visual diagnostics.

Macro-F1 is emphasized because the classification task contains class imbalance and some categories are substantially harder to distinguish than others.

## Results

The final weighted ensemble achieved approximately:

- **Test accuracy:** 0.891
- **Test macro-F1:** 0.856

The best performance was observed for visually distinctive categories, such as Minecraft, DUPLO, Fabuland, and NEXO KNIGHTS.

Lower recall was observed for categories with stronger semantic or visual overlap, including NINJAGO-related categories, LEGO Brand, LEGO Ideas, and Holiday & Event.

The diagnostic analysis suggests that many remaining errors are linked to overlapping commercial categories, ambiguous labels, and visually similar minifigure designs. Grad-CAM visualizations indicate that the model attends to reasonable visual features, but disambiguation remains challenging even for human judgment in certain cases.

## How to Run

The following commands assume that the required image dataset, metadata files, model weights, and embedding files are placed in the expected project directories.

### Run base model training or evaluation
```bash
python "Phase 3 Top 30 trial/top30_train_and_eval_v2.py"
```

### Run the final ensemble with test-time augmentation
```bash
python "Phase 3 ensemble/ensemble_top30.py" --tta --tta-n 8
```

### Generate Grad-CAM heatmaps for misclassified examples
```bash
python "Phase 4 Error diagnostics/extract_misclassified_attention.py"
```

Exact paths may need to be adjusted depending on the local directory structure and dataset location.

## Dependencies

The main dependencies include:

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- XGBoost
- NumPy
- pandas
- matplotlib
- Pillow
- tqdm

A virtual environment is recommended for reproducing the experiments.

## Reproducibility Notes

Random seeds were fixed where possible, including the dataset splitting procedure. The main split used seed=25.

However, exact reproducibility may still vary slightly depending on:

- GPU hardware;
- CUDA version;
- PyTorch version;
- nondeterministic deep learning operations;
- availability of the original image dataset and intermediate embedding files.

The train-validation-test split should be kept fixed when reproducing the reported results.

## Limitations

Several limitations should be noted:

### Dataset Availability
The raw image dataset and large intermediate files are not included in the repository due to file size constraints.

### Commercial Taxonomy Ambiguity
Some official LEGO categories are not visually or semantically clean. This creates cases where two categories may contain highly similar or even overlapping characters.

### Class Imbalance
Although the project focuses on the top 30 categories, class frequency imbalance remains present.

### Model Interpretability
Grad-CAM provides useful visual diagnostics, but it should be interpreted as supporting evidence rather than a strict causal explanation of model decisions.

### Reproducibility Constraints
Full reproduction requires access to the original images, metadata, and in some cases large intermediate files such as embeddings and trained weights.

## Author

**Zhaoyu Wang**
