"""
=============================================================================
File Name: phase1_category_eda_and_grouping.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Performs foundational Exploratory Data Analysis and defines the target classes (Top 30).

Main Functionality/Workflow:
Calculates class frequencies, identifies the long-tail distribution, and establishes the top 30 categories for subsequent classification tasks.

Key Inputs Path:
metadata.json

Key Outputs Path:
Category distribution plots and Top 30 category lists

Important Dependencies:
json, pandas, matplotlib, pathlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 1: Sets the fundamental scope of the classification task by defining the target labels.
=============================================================================
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def relpath(path, base_dir):
    return os.path.relpath(str(Path(path).resolve()), start=str(Path(base_dir).resolve()))


def build_repro_command(args):
    root = project_root()
    cmd = [
        "python",
        relpath(Path(__file__).resolve(), root),
        "--metadata-json",
        relpath(args.metadata_json, root),
        "--out-dir",
        relpath(args.out_dir, root),
        "--threshold",
        str(args.threshold),
        "--other-label",
        args.other_label,
        "--topk-plot",
        str(args.topk_plot),
    ]
    return subprocess.list2cmdline(cmd)


def write_run_log(out_dir, outputs, command):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    log_path = out_dir / "run.log"
    root = project_root()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== {ts} ===\n")
        f.write("cwd: .\n")
        f.write(f"command: {command}\n")
        f.write("outputs:\n")
        for p in outputs:
            f.write(f"- {relpath(p, root)}\n")
        f.write("\n")


def load_metadata(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_counts_csv(path, counter, header_left):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_left, "count"])
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([k, v])


def plot_category_histogram(path, counter, title_prefix, topk=30):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    topk = min(topk, len(counter))
    top = counter.most_common(topk)
    labels = [k for k, _ in top][::-1]
    values = [v for _, v in top][::-1]
    counts = np.array(list(counter.values()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(labels, values, color="#4C72B0")
    axes[0].set_title(f"{title_prefix}: Top {topk} by count")
    axes[0].set_xlabel("Images")
    axes[0].grid(axis="x", alpha=0.2)

    bins = min(50, max(10, int(math.sqrt(len(counts)))))
    axes[1].hist(counts, bins=bins, color="#55A868", edgecolor="white")
    axes[1].set_title(f"{title_prefix}: Distribution of images per category")
    axes[1].set_xlabel("Images per category")
    axes[1].set_ylabel("#Categories")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_path = Path(path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / 'Phase 1 data exploration'),
    )
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--other-label", default="Other")
    parser.add_argument("--topk-plot", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    data = load_metadata(args.metadata_json)

    categories = [d.get("category") for d in data]
    missing = sum(1 for c in categories if not c)
    ctr = Counter(c for c in categories if c)

    print(f"Total records: {len(data)}")
    print(f"Non-empty category: {len(categories) - missing}")
    print(f"Missing/empty category: {missing}")
    print(f"Unique categories: {len(ctr)}")

    out_counts = out_dir / "category_counts.csv"
    write_counts_csv(out_counts, ctr, "category")
    print(f"Wrote: {out_counts.resolve()}")
    outputs.append(out_counts)

    out_hist = out_dir / "category_histogram.png"
    if plot_category_histogram(out_hist, ctr, "Original category", topk=args.topk_plot):
        print(f"Wrote: {out_hist.resolve()}")
        outputs.append(out_hist)
    else:
        print("Plot skipped (matplotlib not available).")

    kept = {c for c, n in ctr.items() if n >= args.threshold}
    grouped = []
    for d in data:
        c = d.get("category")
        grouped.append(c if c in kept else args.other_label)
    ctr_grouped = Counter(grouped)

    print(f"\nThreshold >= {args.threshold}")
    print(f"Kept categories: {len(kept)}")
    print(f"Grouped categories: {len(ctr_grouped)}")
    print(f"Other count: {ctr_grouped.get(args.other_label, 0)} ({ctr_grouped.get(args.other_label, 0) / len(data):.1%})")

    out_group_counts = out_dir / f"category_counts_grouped_threshold_{args.threshold}.csv"
    write_counts_csv(out_group_counts, ctr_grouped, "category_grouped")
    print(f"Wrote: {out_group_counts.resolve()}")
    outputs.append(out_group_counts)

    out_group_hist = out_dir / f"category_grouped_threshold_{args.threshold}_histogram.png"
    if plot_category_histogram(out_group_hist, ctr_grouped, f"Grouped (threshold={args.threshold})", topk=min(args.topk_plot, len(ctr_grouped))):
        print(f"Wrote: {out_group_hist.resolve()}")
        outputs.append(out_group_hist)

    out_rows = []
    for d in data:
        c = d.get("category")
        out_rows.append(
            {
                "id": d.get("id"),
                "minifig_number": d.get("minifig_number"),
                "img_local_path": d.get("img_local_path"),
                "category": c,
                f"category_grouped_threshold_{args.threshold}": c if c in kept else args.other_label,
            }
        )

    out_meta = out_dir / f"metadata_category_grouped_threshold_{args.threshold}.csv"
    with out_meta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote: {out_meta.resolve()}")
    outputs.append(out_meta)
    command = build_repro_command(args)
    write_run_log(out_dir=out_dir, outputs=outputs, command=command)


if __name__ == "__main__":
    main()
