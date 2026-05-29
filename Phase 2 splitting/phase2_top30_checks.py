"""
=============================================================================
File Name: phase2_top30_checks.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Validates the statistical distribution of the Top 30 categories across splits.

Main Functionality/Workflow:
Calculates category proportions in Train/Val/Test and performs statistical checks (e.g., KS-Tests) to ensure identical distributions.

Key Inputs Path:
Train, Val, and Test JSON split files

Key Outputs Path:
Statistical reports and potential distribution visualizations

Important Dependencies:
json, pandas, scipy.stats

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 2: Confirms the stratification quality to prevent distribution shift during training.
=============================================================================
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_split_and_checks import parse_year, parse_eur_value

def get_top_30_categories(train_json_path):
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    counts = defaultdict(int)
    for r in train_data:
        cat = r.get("category")
        if cat:
            counts[cat] += 1
            
    # Get top 30
    top30 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30]
    return set([c[0] for c in top30])

def load_split_data(train_path, val_path, test_path, top30_cats):
    split_obj = {"train": [], "val": [], "test": []}
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for r in json.load(f):
            if r.get("category") in top30_cats:
                split_obj["train"].append(r)
                
    with open(val_path, 'r', encoding='utf-8') as f:
        for r in json.load(f):
            if r.get("category") in top30_cats:
                split_obj["val"].append(r)
                
    with open(test_path, 'r', encoding='utf-8') as f:
        for r in json.load(f):
            if r.get("category") in top30_cats:
                split_obj["test"].append(r)
                
    return split_obj

def write_per_category_ks_csv(out_path, split_obj, min_n=20):
    fields = {
        "year": lambda d: parse_year(d.get("year")),
        "year_released": lambda d: parse_year(d.get("year_released")),
        "current_value_new_eur": lambda d: parse_eur_value(d.get("current_value_new")),
        "current_value_used_eur": lambda d: parse_eur_value(d.get("current_value_used")),
    }

    by_label_split = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for split_name in ["train", "val", "test"]:
        for r in split_obj[split_name]:
            label = r["category"]
            by_label_split[label][split_name].append(r)

    rows = []
    for label, dct in by_label_split.items():
        for field_name, fn in fields.items():
            tr = [fn(r) for r in dct["train"] if fn(r) is not None]
            va = [fn(r) for r in dct["val"] if fn(r) is not None]
            te = [fn(r) for r in dct["test"] if fn(r) is not None]
            if len(tr) < min_n or len(va) < min_n or len(te) < min_n:
                continue
            stat_tv, p_tv = ks_2samp(tr, va)
            stat_tt, p_tt = ks_2samp(tr, te)
            rows.append(
                {
                    "label": label,
                    "field": field_name,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "ks_train_val": float(stat_tv),
                    "p_train_val": float(p_tv),
                    "ks_train_test": float(stat_tt),
                    "p_train_test": float(p_tt),
                    "flag_any_p_lt_0_05": (p_tv < 0.05) or (p_tt < 0.05),
                }
            )

    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["label"])
        w.writeheader()
        w.writerows(rows)

def write_bonferroni_csv(in_path, out_path, alpha=0.05):
    in_path = Path(in_path)
    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["p_train_val"] = float(row["p_train_val"])
            row["p_train_test"] = float(row["p_train_test"])
            rows.append(row)

    m_tests = len(rows) * 2
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "field",
                "n_train",
                "n_val",
                "n_test",
                "p_train_val",
                "p_train_test",
                "p_train_val_bonf",
                "p_train_test_bonf",
                "min_p",
                "min_p_bonf",
                "raw_flag_0_05",
                "bonf_flag",
            ]
        )
        for row in rows:
            pv = float(row["p_train_val"])
            pt = float(row["p_train_test"])
            pv_b = min(1.0, pv * m_tests)
            pt_b = min(1.0, pt * m_tests)
            w.writerow(
                [
                    row["label"],
                    row["field"],
                    row["n_train"],
                    row["n_val"],
                    row["n_test"],
                    pv,
                    pt,
                    pv_b,
                    pt_b,
                    min(pv, pt),
                    min(pv_b, pt_b),
                    (pv < alpha) or (pt < alpha),
                    (pv_b < alpha) or (pt_b < alpha),
                ]
            )

def main():
    base_dir = Path(str(Path(__file__).resolve().parents[1] / 'Phase 2 splitting'))
    train_path = base_dir / "split_metadata_threshold100_seed25_70_15_15_train.json"
    val_path = base_dir / "split_metadata_threshold100_seed25_70_15_15_val.json"
    test_path = base_dir / "split_metadata_threshold100_seed25_70_15_15_test.json"
    
    out_dir = base_dir / "top30_ks_checks"
    out_dir.mkdir(exist_ok=True)
    
    top30_cats = get_top_30_categories(train_path)
    print(f"Found {len(top30_cats)} top categories.")
    
    split_obj = load_split_data(train_path, val_path, test_path, top30_cats)
    
    ks_csv = out_dir / "per_category_ks_top30.csv"
    write_per_category_ks_csv(ks_csv, split_obj)
    print(f"Wrote KS tests to {ks_csv.name}")
    
    bonf_csv = out_dir / "per_category_ks_top30_with_bonferroni.csv"
    write_bonferroni_csv(ks_csv, bonf_csv)
    print(f"Wrote Bonferroni corrected KS tests to {bonf_csv.name}")

if __name__ == "__main__":
    main()
