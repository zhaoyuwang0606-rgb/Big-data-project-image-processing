"""
=============================================================================
File Name: check_themes.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Exploratory data analysis (EDA) to investigate the distribution and overlap of themes within the LEGO metadata.

Main Functionality/Workflow:
Loads the metadata JSON, parses theme lists for each minifigure, and calculates frequency and co-occurrence metrics.

Key Inputs Path:
metadata.json

Key Outputs Path:
Console output detailing theme overlaps

Important Dependencies:
json, collections, pathlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 1: Helps understand the structural complexity of LEGO categorization before model training.
=============================================================================
"""

import json, sys, csv
from collections import defaultdict, Counter
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

data = json.loads(Path('../metadata.json').read_text(encoding='utf-8'))

# ── 1. Theme 与 category 的关系 ──────────────────────────────
theme_to_cats = defaultdict(set)
theme_cat_count = defaultdict(Counter)
cat_multi_theme = defaultdict(int)  # how many minifigs per category have >1 theme

for r in data:
    cat = r.get('category', '')
    themes = r.get('themes') or []
    if not cat:
        continue
    for t in themes:
        theme_to_cats[t].add(cat)
        theme_cat_count[t][cat] += 1
    if len(themes) > 1:
        cat_multi_theme[cat] += 1

# ── 2. Town 的 themes ─────────────────────────────────────────
town_records = [r for r in data if r.get('category') == 'Town']
town_themes = Counter()
town_multi = 0
for r in town_records:
    themes = r.get('themes') or []
    for t in themes:
        town_themes[t] += 1
    if len(themes) > 1:
        town_multi += 1

print('=== Themes inside Town category ===')
print(f'Total Town minifigs: {len(town_records)}')
print(f'Town minifigs with >1 theme: {town_multi} ({town_multi/len(town_records)*100:.1f}%)')
print(f'\nTop themes within Town ({len(town_themes)} unique themes):')
for t, cnt in town_themes.most_common(20):
    cat_count = len(theme_to_cats[t])
    print(f'  {cnt:5d}  "{t}"  [shared by {cat_count} categories]')

# ── 3. Themes that span Town + other categories ───────────────
print('\n=== Town themes that also appear in OTHER categories ===')
cross = [(t, theme_to_cats[t]) for t in town_themes if len(theme_to_cats[t]) > 1]
cross.sort(key=lambda x: -len(x[1]))
for t, cats in cross[:20]:
    other = sorted(cats - {'Town'})
    town_cnt = theme_cat_count[t]['Town']
    print(f'\nTheme: "{t}" (Town has {town_cnt} minifigs, total {len(cats)} categories)')
    for c in other[:8]:
        print(f'    -> [{c}]: {theme_cat_count[t][c]}')
    if len(other) > 8:
        print(f'    -> ... and {len(other)-8} more categories')

# ── 4. Categories with most multi-theme minifigs ─────────────
print('\n\n=== Categories with most multi-theme minifigs ===')
for cat, cnt in sorted(cat_multi_theme.items(), key=lambda x: -x[1])[:15]:
    total = sum(1 for r in data if r.get('category') == cat)
    print(f'  {cnt:5d} / {total:5d}  ({cnt/total*100:5.1f}%)  {cat}')

# ── 5. Overall: how many minifigs have multiple themes ─────────
multi_theme_total = sum(1 for r in data if len(r.get('themes') or []) > 1)
print(f'\nOverall: {multi_theme_total}/{len(data)} minifigs ({multi_theme_total/len(data)*100:.1f}%) have >1 theme')

# ── 6. Plotting Categories with most multi-theme minifigs ─────
import matplotlib.pyplot as plt
import numpy as np

# Find the Top 30 categories by total number of minifigs
cat_totals = Counter(r.get('category') for r in data if r.get('category'))
top_30_cats = [cat for cat, _ in cat_totals.most_common(30)]

# Calculate the percentage of multi-theme minifigs for these top 30 categories
cat_stats = []
for cat in top_30_cats:
    total = cat_totals[cat]
    multi = cat_multi_theme[cat]
    # Skip categories that have 0 multi-theme minifigs (like Minecraft, etc.)
    if multi == 0:
        continue
    pct = (multi / total) * 100 if total > 0 else 0
    cat_stats.append({
        'category': cat,
        'multi_count': multi,
        'total': total,
        'percentage': pct
    })

# Sort them by percentage of multi-theme minifigs (descending)
cat_stats.sort(key=lambda x: x['percentage'], reverse=True)

# Reverse for horizontal bar chart (so highest is at the top)
labels = [x['category'] for x in cat_stats][::-1]
counts = [x['multi_count'] for x in cat_stats][::-1]
totals = [x['total'] for x in cat_stats][::-1]
percentages = [x['percentage'] for x in cat_stats][::-1]

# Dynamic figsize based on remaining categories
fig, ax1 = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))

# Bar chart for percentages (since we are ranking by percentage now)
bars = ax1.barh(labels, percentages, color='#C44E52', alpha=0.7, label='% of Category')
ax1.set_xlabel('Percentage (%) of Minifigs with Multiple Themes', color='#C44E52', fontsize=12)
ax1.tick_params(axis='x', labelcolor='#C44E52')
ax1.set_xlim(0, max(percentages) * 1.1)

# Create a twin axis for raw counts
ax2 = ax1.twiny()
ax2.plot(counts, labels, color='#4C72B0', marker='o', linewidth=2, label='Multi-theme Count')
ax2.set_xlabel('Raw Count of Multi-theme Minifigs', color='#4C72B0', fontsize=12)
ax2.tick_params(axis='x', labelcolor='#4C72B0')
ax2.set_xlim(0, max(counts) * 1.1)

plt.title('Top 30 Categories Ranked by Multi-Theme Percentage (excluding 0%)', fontsize=14, pad=20)
fig.tight_layout()

out_path = Path('themes_overlap_visualization.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'\nVisualization saved to {out_path.resolve()}')

# ── 7. Export Data for GraphPad Prism ─────────────────────────
csv_out_path = Path('prism_themes_overlap.csv')
with csv_out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['Category', 'Percentage (%)', 'Multi-theme Count', 'Total Minifigs'])
    # Since we changed the plot to percentage as primary, we put percentage first in CSV for Prism
    for cat, pct, count, total in zip(labels, percentages, counts, totals):
        writer.writerow([cat, round(pct, 2), count, total])
print(f'Prism Data exported to {csv_out_path.resolve()}')

