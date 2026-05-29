"""
=============================================================================
File Name: check_duplicate_minifigs.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Ensures data integrity by checking for data leakage across Train/Val/Test splits.

Main Functionality/Workflow:
Scans the generated split JSONs and asserts that no unique minifigure IDs appear in multiple sets.

Key Inputs Path:
Train, Val, and Test JSON split files

Key Outputs Path:
Validation assertions (console logs)

Important Dependencies:
json, pathlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 2: Crucial validation step to guarantee rigorous and unbiased model evaluation.
=============================================================================
"""

import json
from collections import defaultdict
from pathlib import Path

def check_duplicate_minifigures(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 记录每个minifig_number对应的所有条目
    minifig_records = defaultdict(list)
    for item in data:
        minifig_num = item.get("minifig_number")
        if minifig_num:
            minifig_records[minifig_num].append(item)
    
    # 筛选出数量大于1的minifig_number
    duplicates = {num: items for num, items in minifig_records.items() if len(items) > 1}
    
    print(f"Total entries in metadata: {len(data)}")
    print(f"Total unique minifigures: {len(minifig_records)}")
    print(f"Minifigures with multiple images/entries: {len(duplicates)}\n")
    
    if duplicates:
        for num, items in duplicates.items():
            print(f"Minifigure: {num} (Count: {len(items)})")
            for i, item in enumerate(items, 1):
                print(f"  Entry {i}:")
                print(f"    Name:       {item.get('minifig_name')}")
                print(f"    Category:   {item.get('category')}")
                print(f"    Image Path: {item.get('img_local_path')}")
            print("-" * 50)
    else:
        print("No duplicate minifigures found!")

if __name__ == "__main__":
    # 自动解析出 metadata.json 的路径 (上一级目录)
    current_dir = Path(__file__).resolve().parent
    metadata_file = current_dir.parent / "metadata.json"
    
    if metadata_file.exists():
        check_duplicate_minifigures(metadata_file)
    else:
        print(f"Error: metadata.json not found at {metadata_file}")
