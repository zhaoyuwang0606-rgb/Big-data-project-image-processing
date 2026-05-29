"""
=============================================================================
File Name: phase2_split_and_checks.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Executes stratified sampling to create Train/Val/Test (70/15/15) splits.

Main Functionality/Workflow:
Groups data by category, performs stratified random splitting ensuring minimum representation, and exports the resulting subsets as JSON files.

Key Inputs Path:
metadata.json

Key Outputs Path:
split_metadata_..._train.json, _val.json, _test.json

Important Dependencies:
json, random, pathlib, collections

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 2: Prepares the standardized datasets utilized by all downstream model training phases.
=============================================================================
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path


_VALUE_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")


def project_root():
    return Path(__file__).resolve().parents[1]


def relpath(path, base_dir):
    return os.path.relpath(str(Path(path).resolve()), start=str(Path(base_dir).resolve()))


def build_repro_command(metadata_json, out_dir, seed, threshold, other_label, train_ratio, val_ratio, test_ratio):
    root = project_root()
    cmd = [
        "python",
        relpath(Path(__file__).resolve(), root),
        "--metadata-json",
        relpath(metadata_json, root),
        "--out-dir",
        relpath(out_dir, root),
        "--seed",
        str(seed),
        "--threshold",
        str(threshold),
        "--other-label",
        other_label,
        "--train",
        str(train_ratio),
        "--val",
        str(val_ratio),
        "--test",
        str(test_ratio),
    ]
    return subprocess.list2cmdline(cmd)


def write_run_log(out_dir, outputs, command, info=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    log_path = out_dir / "run.log"
    root = project_root()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== {ts} ===\n")
        f.write("cwd: .\n")
        f.write(f"command: {command}\n")
        if info:
            f.write(f"summary: {json.dumps(info, ensure_ascii=False)}\n")
        f.write("outputs:\n")
        for p in outputs:
            f.write(f"- {relpath(p, root)}\n")
        f.write("\n")


def parse_year(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if not s or s.lower() == "not known":
        return None
    try:
        return int(s)
    except Exception:
        return None


def parse_eur_value(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "not known":
        return None
    m = _VALUE_RE.search(s)
    if not m:
        return None
    num = m.group(0).replace(",", ".")
    try:
        return float(num)
    except Exception:
        return None


def summarize(nums):
    nums = [x for x in nums if x is not None]
    if not nums:
        return {"n": 0}
    nums_sorted = sorted(nums)
    n = len(nums_sorted)

    def q(p):
        if n == 1:
            return float(nums_sorted[0])
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(nums_sorted[int(k)])
        return float(nums_sorted[f] * (c - k) + nums_sorted[c] * (k - f))

    mean = sum(nums_sorted) / n
    return {
        "n": n,
        "mean": float(mean),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "min": float(nums_sorted[0]),
        "max": float(nums_sorted[-1]),
    }


def compute_grouped_categories(records, threshold, other_label):
    ctr = Counter(r.get("category") for r in records if r.get("category"))
    kept = {c for c, n in ctr.items() if n >= threshold}

    def grouped(c):
        if c and c in kept:
            return c
        return other_label

    return kept, grouped, ctr


def stratified_split_indices(labels, ratios, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    assignment = {}
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        cut1 = n_train
        cut2 = n_train + n_val
        for j in idxs[:cut1]:
            assignment[j] = "train"
        for j in idxs[cut1:cut2]:
            assignment[j] = "val"
        for j in idxs[cut2:]:
            assignment[j] = "test"

    return assignment


def write_category_counts_csv(path, counter, category_field):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([category_field, "count"])
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([k, v])


def write_split_json(out_path, split_records):
    out_path = Path(out_path)
    out_path.write_text(json.dumps(split_records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_lists(out_dir, base_stem, split_obj):
    out_dir = Path(out_dir)
    for key in ["train", "val", "test"]:
        out_path = out_dir / f"{base_stem}_{key}.json"
        out_path.write_text(json.dumps(split_obj[key], ensure_ascii=False, indent=2), encoding="utf-8")


def write_stats_csv(out_path, split_obj):
    fields = {
        "year": lambda d: parse_year(d.get("year")),
        "year_released": lambda d: parse_year(d.get("year_released")),
        "current_value_new_eur": lambda d: parse_eur_value(d.get("current_value_new")),
        "current_value_used_eur": lambda d: parse_eur_value(d.get("current_value_used")),
    }

    rows = []

    def collect(label, split_name, records):
        for field_name, fn in fields.items():
            vals = [fn(r) for r in records]
            missing = sum(1 for v in vals if v is None)
            s = summarize(vals)
            rows.append(
                {
                    "label": label,
                    "split": split_name,
                    "field": field_name,
                    "n_total": len(records),
                    "n_nonmissing": s.get("n", 0),
                    "missing_rate": (missing / len(records)) if records else 0.0,
                    "mean": s.get("mean", ""),
                    "p25": s.get("p25", ""),
                    "p50": s.get("p50", ""),
                    "p75": s.get("p75", ""),
                    "min": s.get("min", ""),
                    "max": s.get("max", ""),
                }
            )

    for split_name in ["train", "val", "test"]:
        collect("__ALL__", split_name, split_obj[split_name])

    labels = sorted({r["category_grouped_threshold_100"] for split_name in ["train", "val", "test"] for r in split_obj[split_name]})
    for label in labels:
        for split_name in ["train", "val", "test"]:
            recs = [r for r in split_obj[split_name] if r["category_grouped_threshold_100"] == label]
            collect(label, split_name, recs)

    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_per_category_ks_csv(out_path, split_obj, min_n=20):
    try:
        from scipy.stats import ks_2samp
    except Exception:
        return False

    fields = {
        "year": lambda d: parse_year(d.get("year")),
        "year_released": lambda d: parse_year(d.get("year_released")),
        "current_value_new_eur": lambda d: parse_eur_value(d.get("current_value_new")),
        "current_value_used_eur": lambda d: parse_eur_value(d.get("current_value_used")),
    }

    by_label_split = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for split_name in ["train", "val", "test"]:
        for r in split_obj[split_name]:
            label = r["category_grouped_threshold_100"]
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

    return True


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

    return {"m_tests": m_tests, "alpha_bonf": alpha / m_tests if m_tests else None}


def plot_distributions_by_category(split_obj, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label_split = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for split in ["train", "val", "test"]:
        for r in split_obj[split]:
            label = r["category_grouped_threshold_100"]
            by_label_split[label][split].append(r)

    labels = sorted(by_label_split.keys())
    split_colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}

    fields = [
        ("year", lambda r: parse_year(r.get("year")), {"kind": "year"}),
        ("year_released", lambda r: parse_year(r.get("year_released")), {"kind": "year"}),
        ("current_value_new_eur", lambda r: parse_eur_value(r.get("current_value_new")), {"kind": "value"}),
        ("current_value_used_eur", lambda r: parse_eur_value(r.get("current_value_used")), {"kind": "value"}),
    ]

    def make_bins(kind, values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        if kind == "year":
            mn, mx = int(min(values)), int(max(values))
            return np.arange(mn - 0.5, mx + 1.5, 1.0)
        vals = np.array([v for v in values if v is not None and v >= 0], dtype=float)
        if len(vals) == 0:
            return None
        logv = np.log1p(vals)
        mn, mx = float(logv.min()), float(logv.max())
        bins = 30
        return np.linspace(mn, mx, bins + 1)

    for field_name, fn, cfg in fields:
        all_vals = []
        for label in labels:
            for split in ["train", "val", "test"]:
                for r in by_label_split[label][split]:
                    all_vals.append(fn(r))
        bins = make_bins(cfg["kind"], all_vals)

        n = len(labels)
        cols = 6
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.8), sharex=True)
        axes = np.array(axes).reshape(rows, cols)

        for idx, label in enumerate(labels):
            ax = axes[idx // cols, idx % cols]
            for split in ["train", "val", "test"]:
                vals = [fn(r) for r in by_label_split[label][split]]
                vals = [v for v in vals if v is not None]
                if cfg["kind"] == "value":
                    vals = [math.log1p(v) for v in vals if v >= 0]
                if not vals:
                    continue
                ax.hist(vals, bins=bins, alpha=0.35, color=split_colors[split], density=True)
            ax.set_title(label, fontsize=10)
            ax.grid(axis="y", alpha=0.15)

        for j in range(n, rows * cols):
            axes[j // cols, j % cols].axis("off")

        xlabel = f"log1p({field_name})" if cfg["kind"] == "value" else field_name
        handles = [plt.Line2D([0], [0], color=split_colors[s], lw=6, alpha=0.7) for s in ["train", "val", "test"]]
        fig.legend(handles, ["train", "val", "test"], loc="upper right", frameon=True)

        fig.suptitle(f"Distribution by category (overlay train/val/test): {field_name}", y=0.995, fontsize=14)
        fig.supxlabel(xlabel)
        fig.supylabel("Density")
        fig.tight_layout(rect=[0, 0, 0.98, 0.97])

        out_path = out_dir / f"dist_by_category_{field_name}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return True


def run(metadata_json, out_dir, seed, threshold, other_label, train_ratio, val_ratio, test_ratio):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    with Path(metadata_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    kept, grouped_fn, ctr = compute_grouped_categories(data, threshold=threshold, other_label=other_label)

    grouped_labels = [grouped_fn(r.get("category")) for r in data]
    assignment = stratified_split_indices(grouped_labels, (train_ratio, val_ratio, test_ratio), seed=seed)

    split_obj = {"train": [], "val": [], "test": []}
    for i, r in enumerate(data):
        out = dict(r)
        out["category_grouped_threshold_100"] = grouped_fn(r.get("category"))
        out["split"] = assignment[i]
        split_obj[assignment[i]].append(out)

    out_base = f"split_metadata_threshold{threshold}_seed{seed}_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}"

    out_json = out_dir / f"{out_base}.json"
    out_json.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(out_json)

    write_split_lists(out_dir, out_base, split_obj)
    outputs.extend([out_dir / f"{out_base}_{k}.json" for k in ["train", "val", "test"]])

    grouped_ctr = Counter(r["category_grouped_threshold_100"] for r in split_obj["train"] + split_obj["val"] + split_obj["test"])
    out_group_counts = out_dir / f"category_counts_grouped_threshold_{threshold}.csv"
    write_category_counts_csv(out_group_counts, grouped_ctr, "category_grouped")
    outputs.append(out_group_counts)
    out_counts = out_dir / "category_counts.csv"
    write_category_counts_csv(out_counts, ctr, "category")
    outputs.append(out_counts)

    stats_csv = out_dir / f"split_stats_threshold{threshold}_seed{seed}_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}.csv"
    write_stats_csv(stats_csv, split_obj)
    outputs.append(stats_csv)

    ks_csv = out_dir / f"per_category_ks_threshold{threshold}_seed{seed}_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}.csv"
    have_ks = write_per_category_ks_csv(ks_csv, split_obj)
    if have_ks:
        outputs.append(ks_csv)

    bonf_info = None
    if have_ks:
        bonf_csv = out_dir / f"per_category_ks_threshold{threshold}_seed{seed}_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}_with_bonferroni.csv"
        bonf_info = write_bonferroni_csv(ks_csv, bonf_csv)
        outputs.append(bonf_csv)

    plot_distributions_by_category(split_obj, out_dir)
    outputs.extend(
        [
            out_dir / "dist_by_category_year.png",
            out_dir / "dist_by_category_year_released.png",
            out_dir / "dist_by_category_current_value_new_eur.png",
            out_dir / "dist_by_category_current_value_used_eur.png",
        ]
    )

    info = {
        "out_json": relpath(out_json, project_root()),
        "stats_csv": relpath(stats_csv, project_root()),
        "ks_csv": relpath(ks_csv, project_root()) if have_ks else None,
        "bonferroni": bonf_info,
        "kept_categories": len(kept),
        "grouped_categories": len(grouped_ctr),
        "other_count": grouped_ctr.get(other_label, 0),
        "total": len(data),
    }
    command = build_repro_command(metadata_json, out_dir, seed, threshold, other_label, train_ratio, val_ratio, test_ratio)
    write_run_log(out_dir=out_dir, outputs=outputs, command=command, info=info)

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--other-label", default="Other")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    info = run(
        metadata_json=args.metadata_json,
        out_dir=args.out_dir,
        seed=args.seed,
        threshold=args.threshold,
        other_label=args.other_label,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
