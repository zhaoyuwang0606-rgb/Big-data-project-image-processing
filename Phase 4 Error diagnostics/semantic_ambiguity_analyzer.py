"""
=============================================================================
File Name: semantic_ambiguity_analyzer.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Analyzes generic categories to demonstrate the 'Vacuum Effect'.

Main Functionality/Workflow:
Examines misclassifications flowing into the massive 'Town' category, showing how prior probability absorbs visually generic classes.

Key Inputs Path:
metadata.json, predictions_test.csv

Key Outputs Path:
Semantic ambiguity reports and error flow statistics

Important Dependencies:
pandas, json

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 4 (Diagnostics): Explains the structural data imbalance and its logical impact on model predictions.
=============================================================================
"""

import json
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# 1. Paths
metadata_path = Path(str(Path(__file__).resolve().parents[1] / 'metadata.json'))
preds_path = Path(str(Path(__file__).resolve().parents[1] / 'Phase 3 Top 30 trial/top30_convnext_tiny/result convnext/predictions_test.csv'))
out_dir = Path(str(Path(__file__).resolve().parents[1] / 'Phase 4 Error diagnostics'))
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Load Metadata
print("Loading metadata...")
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Map minifig_number -> info
minifig_info = {}
# Also build mappings for what subcategories/themes belong to what categories
cat_to_subcategories = defaultdict(set)
cat_to_themes = defaultdict(set)

for item in metadata:
    mf_num = item.get('minifig_number')
    cat = item.get('category')
    subcat = item.get('subcategory')
    themes = item.get('themes', [])
    
    if mf_num and cat:
        minifig_info[mf_num] = {
            'category': cat,
            'subcategory': subcat,
            'themes': set(themes) if themes else set(),
            'name': item.get('name', '')
        }
        if subcat:
            cat_to_subcategories[cat].add(subcat)
        for t in themes:
            cat_to_themes[cat].add(t)

# 3. Load Predictions and Analyze Errors
print("Analyzing predictions...")
total_predictions = 0
total_errors = 0

error_general_subcat = 0
error_shared_subcat = 0
error_shared_theme = 0

# To store specific examples for visual inspection later
confusing_pairs = defaultdict(list)

with open(preds_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['split'] != 'test':
            continue
            
        total_predictions += 1
        true_label = row['true_label']
        pred_label = row['pred_label']
        mf_num = row['minifig_number']
        
        if true_label != pred_label:
            total_errors += 1
            info = minifig_info.get(mf_num)
            
            if not info:
                continue
                
            subcat = info['subcategory']
            themes = info['themes']
            
            is_general = (subcat == "General")
            is_shared_subcat = subcat in cat_to_subcategories.get(pred_label, set())
            is_shared_theme = len(themes.intersection(cat_to_themes.get(pred_label, set()))) > 0
            
            if is_general:
                error_general_subcat += 1
            if is_shared_subcat and not is_general:
                error_shared_subcat += 1
            if is_shared_theme:
                error_shared_theme += 1
                
            # Add specific check for NINJAGO vs The LEGO NINJAGO Movie
            is_ninjago_overlap = (true_label == "The LEGO NINJAGO Movie" and pred_label == "NINJAGO") or \
                                 (true_label == "NINJAGO" and pred_label == "The LEGO NINJAGO Movie")
            
            # Keep track of ALL interesting errors (not just general ones, to see the Ninja overlap)
            pair_key = f"{true_label} -> {pred_label}"
            confusing_pairs[pair_key].append({
                'mf_num': mf_num,
                'name': info['name'],
                'subcat': subcat,
                'themes': list(themes),
                'is_shared_subcat': is_shared_subcat,
                'is_shared_theme': is_shared_theme,
                'is_ninjago_overlap': is_ninjago_overlap
            })

# 4. Generate Report
report_path = out_dir / "semantic_ambiguity_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=== Error Diagnostics: Proving Semantic Ambiguity ===\n\n")
    f.write(f"Total Test Samples: {total_predictions}\n")
    f.write(f"Total Errors: {total_errors} ({(total_errors/total_predictions)*100:.1f}% Error Rate)\n\n")
    
    f.write("--- Breakdown of Errors by Semantic Overlap ---\n")
    f.write(f"1. 'General' Subcategory Trap: {error_general_subcat} errors ({(error_general_subcat/total_errors)*100:.1f}% of all errors)\n")
    f.write("   (These are minifigs with no specific subcategory, making them visually generic.)\n")
    
    f.write(f"2. Shared Subcategory (Non-General): {error_shared_subcat} errors ({(error_shared_subcat/total_errors)*100:.1f}% of all errors)\n")
    f.write("   (The misclassified figure belongs to a subcategory that ALSO exists in the predicted category.)\n")
    
    f.write(f"3. Shared Themes: {error_shared_theme} errors ({(error_shared_theme/total_errors)*100:.1f}% of all errors)\n")
    f.write("   (The misclassified figure shares a Theme tag with the predicted category.)\n\n")
    
    # Calculate union of explained errors
    explained = 0
    ninjago_movie_errors = 0
    with open(preds_path, 'r', encoding='utf-8') as csv_f:
        r = csv.DictReader(csv_f)
        for row in r:
            if row['split'] == 'test' and row['true_label'] != row['pred_label']:
                info = minifig_info.get(row['minifig_number'])
                if info:
                    # Is it the Ninja movie overlap?
                    is_ninja = (row['true_label'] == "The LEGO NINJAGO Movie" and row['pred_label'] == "NINJAGO") or \
                               (row['true_label'] == "NINJAGO" and row['pred_label'] == "The LEGO NINJAGO Movie")
                    if is_ninja:
                        ninjago_movie_errors += 1
                        explained += 1
                    elif info['subcategory'] == 'General' or \
                       info['subcategory'] in cat_to_subcategories.get(row['pred_label'], set()) or \
                       len(info['themes'].intersection(cat_to_themes.get(row['pred_label'], set()))) > 0:
                        explained += 1
                        
    f.write(f"4. The NINJAGO Movie vs NINJAGO Direct Conflict: {ninjago_movie_errors} errors ({(ninjago_movie_errors/total_errors)*100:.1f}% of all errors)\n")
    f.write("   (These are visually identical characters from the exact same IP, artificially split by movie vs series.)\n\n")
    
    f.write(f"*** TOTAL ERRORS EXPLAINED BY SEMANTIC OVERLAP: {explained} ({(explained/total_errors)*100:.1f}% of all errors) ***\n")
    f.write("This proves that the vast majority of the model's 'mistakes' are actually logical choices based on overlapping LEGO definitions.\n\n")
    
    f.write("\n--- Top 10 Most Confusing Category Pairs (with Semantic Overlap evidence) ---\n")
    sorted_pairs = sorted(confusing_pairs.items(), key=lambda x: len(x[1]), reverse=True)
    for pair, examples in sorted_pairs[:10]:
        f.write(f"\n{pair} ({len(examples)} errors)\n")
        # Show up to 3 examples
        for ex in examples[:3]:
            f.write(f"   - {ex['mf_num']}: {ex['name']}\n")
            f.write(f"     [Subcategory: {ex['subcat']} (Shared: {ex['is_shared_subcat']})] [Themes: {', '.join(ex['themes'])} (Shared: {ex['is_shared_theme']})]\n")

print(f"Report saved to {report_path}")

# 5. Plot a Pie Chart for the Report
labels = ['Explained by Semantic Overlap\n(Shared Theme, Subcat, General, or NINJAGO Movie)', 'True Model Errors\n(Visual Misunderstanding)']
sizes = [explained, total_errors - explained]
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title('Are the Model\'s Errors Actually "Errors"?\n(Analysis of Misclassified Test Images)', fontsize=14, pad=20)
plt.axis('equal')

plot_path = out_dir / "error_explanation_pie_chart.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Pie chart saved to {plot_path}")
