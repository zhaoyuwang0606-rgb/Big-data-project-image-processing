"""
=============================================================================
File Name: plot_normalized_cm.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Generates normalized confusion matrices for ensemble results.

Main Functionality/Workflow:
Reads ensemble prediction logs and uses matplotlib/seaborn to render high-quality normalized confusion matrices.

Key Inputs Path:
Ensemble prediction logs/CSVs

Key Outputs Path:
Normalized confusion matrix PNGs

Important Dependencies:
matplotlib, seaborn, pandas, sklearn

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 3 (Ensemble): Visualizes the final model performance for reporting and diagnostics.
=============================================================================
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def plot_normalized_confusion_matrix_from_log(log_path):
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: Log file {log_file} does not exist.")
        return

    print(f"Parsing log file: {log_file.name} ...")
    log_text = log_file.read_text(encoding="utf-8")
    
    # Load labels from a canonical label map
    label_map_path = Path(str(Path(__file__).resolve().parents[2] / 'Phase 3 Top 30 trial/top30_resnet 50/label_map.json'))
    labels = []
    if label_map_path.exists():
        with open(label_map_path, 'r', encoding='utf-8') as f:
            lm = json.load(f)
            labels = lm.get("labels", [])
    
    # We look for the matrix blocks between 'TRAIN Confusion Matrix:' (or TEST) and '==='
    for split in ["TRAIN", "TEST"]:
        pattern = rf"{split} Confusion Matrix:\n(.*?)(?:\n===|\Z)"
        match = re.search(pattern, log_text, re.DOTALL)
        if match:
            matrix_str = match.group(1)
            # Remove brackets to get just numbers separated by whitespace
            clean_text = matrix_str.replace('[', ' ').replace(']', ' ')
            try:
                numbers = [int(x) for x in clean_text.split()]
                if not numbers:
                    continue
                
                # Reshape to NxN square matrix
                size = int(np.sqrt(len(numbers)))
                cm = np.array(numbers).reshape(size, size)
                
                # Normalize the confusion matrix over the true (row) labels
                # Add a small epsilon to avoid division by zero if a row sum is 0
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_normalized = cm / (row_sums + 1e-8)
                
                # Plot the colored normalized heatmap
                plt.figure(figsize=(18, 16))
                
                # If labels length matches the size, use them, otherwise use numbers
                tick_labels = labels if len(labels) == size else "auto"
                
                # Using format ".2f" for floats and a colormap appropriate for standardized values (0 to 1)
                sns.heatmap(
                    cm_normalized, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="Blues", 
                    cbar=True, 
                    vmin=0.0, 
                    vmax=1.0,
                    xticklabels=tick_labels,
                    yticklabels=tick_labels
                )
                plt.title(f"{split} Normalized Confusion Matrix (Ver 2)")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                
                # Rotate x-axis labels for better readability if we have text labels
                if len(labels) == size:
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                
                out_path = log_file.parent / f"{split.lower()}_normalized_confusion_matrix_v2.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Successfully saved {split} normalized confusion matrix to {out_path}")
            except Exception as e:
                print(f"Failed to parse {split} CM: {e}")
        else:
            print(f"Could not find {split} Confusion Matrix block in the log.")

if __name__ == "__main__":
    logs_to_process = [
        str(Path(__file__).resolve().parents[2] / 'ensemble_logs/ensemble_base_clip_resnet50_vit_b_16_dinov2_tta8/run_20260426_152441.log'),
        str(Path(__file__).resolve().parents[2] / 'ensemble_logs/ensemble_base_clip_resnet50_vit_b_16_dinov2_notta/run_20260426_152440.log')
    ]
    
    # Allow passing arguments via command line
    if len(sys.argv) > 1:
        logs_to_process = sys.argv[1:]
        
    for log_file in logs_to_process:
        plot_normalized_confusion_matrix_from_log(log_file)
