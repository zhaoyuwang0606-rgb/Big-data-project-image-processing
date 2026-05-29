"""
=============================================================================
File Name: check_town_overlap.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Investigates semantic overlap specifically involving the 'Town' category to identify early signs of the Vacuum Effect.

Main Functionality/Workflow:
Filters metadata for 'Town' entries and compares their features (tags, subcategories) against visually similar classes.

Key Inputs Path:
metadata.json

Key Outputs Path:
Console output / textual analysis of Town overlaps

Important Dependencies:
json, pathlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 1: Preemptively identifies structural noise in the largest category.
=============================================================================
"""

import json, sys, csv
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

data = json.loads(Path('../metadata.json').read_text(encoding='utf-8'))

# Build: subcategory -> set of categories
sub_to_cats = defaultdict(set)
sub_to_counts = defaultdict(lambda: defaultdict(int))

for r in data:
    cat = r.get('category', '')
    sub = r.get('subcategory', '')
    if cat and sub:
        sub_to_cats[sub].add(cat)
        sub_to_counts[sub][cat] += 1

# Find subcategories shared between Town and at least one other category
print('=== Subcategories shared between Town and other categories ===\n')
shared = []
for sub, cats in sub_to_cats.items():
    if 'Town' in cats and len(cats) > 1:
        shared.append((sub, cats))

if not shared:
    print('No overlap found - Town subcategories are unique to Town.')
else:
    shared.sort(key=lambda x: x[0])
    for sub, cats in shared:
        print(f'Subcategory: "{sub}"')
        for c in sorted(cats):
            print(f'  [{c}]: {sub_to_counts[sub][c]} minifigs')
        print()

print(f'Total overlapping subcategories: {len(shared)}')

# Also show subcategories that are ambiguous across ANY categories (not just Town)
print('\n\n=== All subcategories appearing in 2+ different categories ===\n')
all_shared = [(sub, cats) for sub, cats in sub_to_cats.items() if len(cats) > 1]
all_shared.sort(key=lambda x: -len(x[1]))
for sub, cats in all_shared[:30]:
    print(f'"{sub}" -> appears in {len(cats)} categories: {sorted(cats)}')

# ── Visualization: Heatmap of Ambiguous Subcategories ──
import matplotlib.pyplot as plt
import numpy as np

# Get top 20 most ambiguous subcategories (by number of categories they appear in)
top_ambig = all_shared[:20]
# Rename long subcategory/category names for better display
sub_labels = []
for x in top_ambig:
    label = x[0]
    if "The Hobbit and The Lord of the Rings" in label:
        label = "The Lord of the Rings"
    sub_labels.append(label)

# Collect categories that these subcategories appear in
# Count how many minifigs are in each category across these top subcategories
cat_frequency = defaultdict(int)
for sub, cats in top_ambig:
    for cat in cats:
        # Also rename the category if it matches the long name
        cat_name = "The Lord of the Rings" if cat == "The Hobbit and The Lord of the Rings" else cat
        cat_frequency[cat_name] += sub_to_counts[sub][cat]

# Only keep the top 30 categories with the most minifigs in these ambiguous subcategories
top_cats = sorted(cat_frequency.items(), key=lambda x: -x[1])[:30]
cat_labels = sorted([x[0] for x in top_cats])

print(f'\n[DEBUG] The heatmap X-axis contains {len(cat_labels)} categories.')

# Build heatmap data matrix
heatmap_data = np.zeros((len(sub_labels), len(cat_labels)))
for i, (original_sub, _) in enumerate(top_ambig):
    for j, cat_name in enumerate(cat_labels):
        # We need to map the clean category name back to the original to get counts
        original_cat = "The Hobbit and The Lord of the Rings" if cat_name == "The Lord of the Rings" else cat_name
        if original_cat in sub_to_cats[original_sub]:
            heatmap_data[i, j] = sub_to_counts[original_sub][original_cat]

fig, ax = plt.subplots(figsize=(12, 8))

# Mask zeros so they appear blank
masked_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)
cmap = plt.cm.YlOrRd
cmap.set_bad(color='white')

im = ax.imshow(masked_data, cmap=cmap, aspect='auto')

# Show all ticks and label them
ax.set_xticks(np.arange(len(cat_labels)))
ax.set_yticks(np.arange(len(sub_labels)))
ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(sub_labels, fontsize=10)

# Removed text annotations inside the heatmap cells for a cleaner look

ax.set_title("Heatmap of Cross-Category Subcategories\n(Top 20 Subcategories vs Top 30 Categories by Overlap)", fontsize=14, pad=20)
fig.tight_layout()

out_path = Path('subcategory_overlap_heatmap.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'\nVisualization saved to {out_path.resolve()}')

# ── Export Data for GraphPad Prism ───────────────────────────
csv_out_path = Path('prism_subcategory_heatmap.csv')
with csv_out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Header: First column empty (for row names), then category names
    writer.writerow(['Subcategory'] + cat_labels)
    # Write rows
    for i, sub in enumerate(sub_labels):
        row = [sub]
        for j in range(len(cat_labels)):
            val = int(heatmap_data[i, j])
            row.append(val if val > 0 else '') # Prism handles empty cells well for heatmaps
        writer.writerow(row)
print(f'Prism Data exported to {csv_out_path.resolve()}')

