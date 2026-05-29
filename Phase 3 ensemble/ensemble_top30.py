"""
=============================================================================
File Name: ensemble_top30.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Executes the final weighted ensemble and Test-Time Augmentation (TTA).

Main Functionality/Workflow:
Combines logits from fine-tuned CV models and probabilities from XGBoost, optionally applies 8-crop TTA, and evaluates the final predictions.

Key Inputs Path:
Trained CV models (.pt), Trained XGBoost models (.ubj), Images

Key Outputs Path:
Ensemble logs, test predictions, normalized confusion matrices

Important Dependencies:
torch, xgboost, numpy, scikit-learn

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 3 (Ensemble): Represents the peak predictive capability of the project, pushing classification to its limits.
=============================================================================
"""

"""
Ensemble script (Optional TTA): Top-30 ResNet-50 + ViT-B/16 + ConvNeXt-Tiny + XGB multimodel
- Evaluates train, val, and test splits in a single run.
- Tunes ensemble weights on VAL split.
- Applies chosen weights to train, val, and test splits.
- Reports Macro F1, Accuracy, and Confusion Matrix (for train and test).
"""
import argparse
import json
import random
import sys
import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch import nn
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    resnet50,
    vit_b_16,
)

# ─────────────────────── PATHS ─────────────────────────────
BASE = Path(__file__).resolve().parents[1]

RESNET_DIR   = BASE / "Phase 3 Top 30 trial" / "top30_resnet 50"
VIT_DIR      = BASE / "Phase 3 Top 30 trial" / "top30_vit"
CONVNEXT_DIR = BASE / "Phase 3 Top 30 trial" / "top30_convnext_tiny"

RESNET_CKPT = RESNET_DIR / "best.pt"
VIT_CKPT = VIT_DIR / "best.pt"
CONVNEXT_CKPT = CONVNEXT_DIR / "best.pt"

RESNET_LABEL_MAP = RESNET_DIR / "label_map.json"
VIT_LABEL_MAP = VIT_DIR / "Result vit" / "label_map.json"
CONVNEXT_LABEL_MAP = CONVNEXT_DIR / "result convnext" / "label_map.json"

SPLITS_DIR = BASE / "Phase 2 splitting"
TRAIN_JSON = SPLITS_DIR / "split_metadata_threshold100_seed25_70_15_15_train.json"
VAL_JSON = SPLITS_DIR / "split_metadata_threshold100_seed25_70_15_15_val.json"
TEST_JSON = SPLITS_DIR / "split_metadata_threshold100_seed25_70_15_15_test.json"
IMAGE_ROOT   = BASE / "images" / "images"
# ───────────────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def resolve_image_path(img_local_path, image_root, minifig_number=None):
    image_root = Path(image_root)
    name = Path(img_local_path).name if img_local_path else ""
    stem = Path(name).stem if name else ""
    candidates = []
    if minifig_number:
        ms = str(minifig_number)
        candidates += [image_root / f"{ms}.jpg", image_root / f"{ms}.png",
                       image_root / f"{ms.lower()}.jpg", image_root / f"{ms.lower()}.png"]
    if name:
        candidates += [image_root / name, image_root / f"{stem}.jpg",
                       image_root / f"{stem}.png"]
    for p in candidates:
        if p.exists():
            return p
    return None

def build_resnet(num_classes):
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    return model

def build_vit(num_classes):
    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    return model

def build_convnext_tiny(num_classes):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model

def get_transforms(model_name, tta_strength="default"):
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
    mean, std = weights.transforms().mean, weights.transforms().std

    det = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if tta_strength == "strong":
        color = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08)
        scale = (0.65, 1.0)
        rot = transforms.RandomRotation(degrees=15)
    else:
        color = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        scale = (0.7, 1.0)
        rot = transforms.RandomRotation(degrees=0)

    tta = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        rot,
        color,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return det, tta

@torch.no_grad()
def infer(model, pil_images, labels, device, transform, tta_n=1, batch_size=32):
    model.eval()
    all_probs = []
    all_y = []

    try:
        from tqdm import tqdm
        batch_iter = tqdm(range(0, len(pil_images), batch_size), desc="Inference")
    except ImportError:
        batch_iter = range(0, len(pil_images), batch_size)

    for start in batch_iter:
        batch_imgs = pil_images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        pass_probs = []
        for _ in range(tta_n):
            tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pass_probs.append(probs)

        avg_probs = np.mean(pass_probs, axis=0)
        all_probs.append(avg_probs)
        all_y.extend(batch_labels)

    return np.concatenate(all_probs), np.array(all_y)

def align_probs_to_canonical(probs, model_label_to_idx, canonical_labels):
    reorder = []
    for lab in canonical_labels:
        if lab not in model_label_to_idx:
            raise ValueError(f"Label {lab!r} missing from a model label map.")
        reorder.append(model_label_to_idx[lab])
    return probs[:, reorder]

def simplex_grid(num_models, step):
    n = round(1.0 / step)
    if not np.isclose(n * step, 1.0):
        raise ValueError(f"--step must evenly divide 1.0, got step={step}.")
    if num_models < 2:
        raise ValueError("num_models must be >= 2")

    def rec(prefix, remaining, k):
        if k == 1:
            yield prefix + [remaining]
            return
        for i in range(remaining + 1):
            yield from rec(prefix + [i], remaining - i, k - 1)

    for ints in rec([], n, num_models):
        yield [i / n for i in ints]

def clean_price(price_str):
    if not price_str or price_str == "Not known":
        return np.nan
    s = str(price_str)
    keep = []
    dot_used = False
    for ch in s:
        if ch.isdigit():
            keep.append(ch)
        elif ch == "." and not dot_used:
            keep.append(ch)
            dot_used = True
    if not keep:
        return np.nan
    try:
        return float("".join(keep))
    except ValueError:
        return np.nan

def clean_year(year_str):
    if not year_str or str(year_str) == "Not known":
        return np.nan
    try:
        return float(year_str)
    except ValueError:
        return np.nan

def build_metadata_dict(metadata_list):
    meta_dict = {}
    for r in metadata_list:
        path = r.get("img_local_path")
        if not path:
            continue
        y_rel = clean_year(r.get("year_released"))
        p_new = clean_price(r.get("current_value_new"))
        p_used = clean_price(r.get("current_value_used"))
        meta_dict[path] = [y_rel, p_new, p_used]
    return meta_dict

def load_embeddings_npz(npz_path):
    data = np.load(npz_path)
    emb_array = data["embeddings"]
    fn_array = data["filenames"]
    emb_dict = {str(fn): emb for fn, emb in zip(fn_array, emb_array)}
    return emb_dict

def load_xgb_label_to_idx(label_map_path):
    idx_to_label = load_json(label_map_path)
    label_to_idx = {}
    for k, v in idx_to_label.items():
        label_to_idx[str(v)] = int(k)
    return label_to_idx

def build_xgb_probs(records, metadata_json, embeddings_npz, xgb_model_path, xgb_label_map, canonical_labels):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError("xgboost is required to include multimodel XGB outputs.") from e

    meta = load_json(metadata_json)
    meta_dict = build_metadata_dict(meta)
    emb_dict = load_embeddings_npz(embeddings_npz)

    X = []
    for r in records:
        path = r.get("img_local_path")
        filename = Path(path).name if path else ""
        v_feat = emb_dict[filename]
        t_feat = meta_dict.get(path, [np.nan, np.nan, np.nan])
        X.append(np.concatenate([v_feat.astype(np.float32), np.array(t_feat, dtype=np.float32)], axis=0))
    X = np.stack(X, axis=0)

    clf = xgb.XGBClassifier()
    clf.load_model(xgb_model_path)
    probs = clf.predict_proba(X)

    xgb_label_to_idx = load_xgb_label_to_idx(xgb_label_map)
    probs = align_probs_to_canonical(probs, xgb_label_to_idx, canonical_labels)
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05, help="Simplex grid step (must divide 1.0).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta", action="store_true", help="Enable TTA for CNN/ViT models.")
    parser.add_argument("--tta-n", type=int, default=8, help="Number of augmented passes per image if --tta is set.")
    parser.add_argument("--tta-strength", choices=["default", "strong"], default="default")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--xgb", type=str, default="", help="Comma-separated: clip,resnet50,vit_b_16,dinov2")
    parser.add_argument("--metadata-json", type=str, default=str(BASE / "metadata.json"))
    parser.add_argument("--embeddings-root", type=str, default=str(BASE / "Phase 3 multimodel" / "Step 1 embedding"))
    parser.add_argument("--xgb-results-root", type=str, default=str(BASE / "Phase 3 multimodel" / "Step 2 ML classifier" / "results"))
    args = parser.parse_args()

    # Set up logging dynamically
    xgb_name = args.xgb.replace(',', '_') if args.xgb else "no_xgb"
    tta_name = f"tta{args.tta_n}" if args.tta else "notta"
    folder_name = f"ensemble_base_{xgb_name}_{tta_name}"
    log_dir = BASE / "Phase 3 ensemble" / "ensemble_logs" / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = log_dir / f"run_{timestamp}.log"
    
    sys.stdout = TeeLogger(log_filepath)
    print(f"=== Ensemble Run Started at {timestamp} ===")
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Log saved to: {log_filepath}\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"TTA Enabled: {args.tta} (Passes: {args.tta_n if args.tta else 1})")

    # Load labels
    lm_r = load_json(RESNET_LABEL_MAP)
    canonical_labels = lm_r["labels"]
    canonical_label_to_idx = lm_r["label_to_idx"]
    num_classes = len(canonical_labels)
    lm_v = load_json(VIT_LABEL_MAP)
    lm_c = load_json(CONVNEXT_LABEL_MAP)

    # Determine XGB config and valid intersection
    xgb_specs = []
    xgb_arg = args.xgb.strip()
    if xgb_arg:
        for name in [s.strip() for s in xgb_arg.split(",") if s.strip()]:
            xgb_specs.append(name)

    xgb_paths = {}
    available_xgb = None
    if xgb_specs:
        results_root = Path(args.xgb_results_root)
        embeddings_root = Path(args.embeddings_root)
        xgb_paths = {
            "clip": (
                results_root / "xgb clip" / "clip" / "xgb_model.ubj",
                results_root / "xgb clip" / "clip" / "label_map.json",
                embeddings_root / "embeddings_clip" / "all_features.npz",
            ),
            "resnet50": (
                results_root / "xgb resnet_50" / "resnet50" / "xgb_model.ubj",
                results_root / "xgb resnet_50" / "resnet50" / "label_map.json",
                embeddings_root / "embeddings_resnet50" / "all_features.npz",
            ),
            "vit_b_16": (
                results_root / "xgb_vit" / "vit_b_16" / "xgb_model.ubj",
                results_root / "xgb_vit" / "vit_b_16" / "label_map.json",
                embeddings_root / "embeddings_vit_b_16" / "all_features.npz",
            ),
            "dinov2": (
                results_root / "xgb_dino" / "dinov2" / "xgb_model.ubj",
                results_root / "xgb_dino" / "dinov2" / "label_map.json",
                embeddings_root / "embeddings_dinov2" / "all_features.npz",
            ),
        }
        for name in xgb_specs:
            if name not in xgb_paths:
                raise ValueError(f"Unknown --xgb model: {name}")
            _, _, emb_npz = xgb_paths[name]
            data = np.load(emb_npz)
            fns = set(str(fn) for fn in data["filenames"])
            available_xgb = fns if available_xgb is None else (available_xgb & fns)

    def load_split_data(json_path):
        records_all = load_json(json_path)
        valid_cats = set(canonical_labels)
        records = [
            r for r in records_all
            if r.get("category") in valid_cats
            and resolve_image_path(r.get("img_local_path"), IMAGE_ROOT, r.get("minifig_number")) is not None
        ]
        if available_xgb is not None:
            records = [r for r in records if Path(r.get("img_local_path") or "").name in available_xgb]
        
        pil_images = []
        y_true = []
        for r in records:
            path = resolve_image_path(r.get("img_local_path"), IMAGE_ROOT, r.get("minifig_number"))
            pil_images.append(Image.open(path).convert("RGB"))
            y_true.append(canonical_label_to_idx[r["category"]])
        return records, pil_images, np.array(y_true)

    # Load all splits
    print("\nLoading data for train, val, and test splits...")
    train_records, train_imgs, y_train = load_split_data(TRAIN_JSON)
    val_records, val_imgs, y_val = load_split_data(VAL_JSON)
    test_records, test_imgs, y_test = load_split_data(TEST_JSON)

    print(f"Train samples: {len(train_records)}")
    print(f"Val samples:   {len(val_records)}")
    print(f"Test samples:  {len(test_records)}")

    # Transforms
    det_r, tta_r = get_transforms("resnet50", tta_strength=args.tta_strength)
    det_v, tta_v = get_transforms("vit_b_16", tta_strength=args.tta_strength)
    det_c, tta_c = get_transforms("convnext_tiny", tta_strength=args.tta_strength)
    
    tf_r = tta_r if args.tta else det_r
    tf_v = tta_v if args.tta else det_v
    tf_c = tta_c if args.tta else det_c
    tta_n = args.tta_n if args.tta else 1

    # Load models
    print("\nLoading models into memory...")
    model_r = build_resnet(num_classes).to(device)
    model_r.load_state_dict(torch.load(RESNET_CKPT, map_location=device)["model"])

    model_v = build_vit(num_classes).to(device)
    model_v.load_state_dict(torch.load(VIT_CKPT, map_location=device)["model"])

    model_c = build_convnext_tiny(num_classes).to(device)
    model_c.load_state_dict(torch.load(CONVNEXT_CKPT, map_location=device)["model"])

    model_names = ["resnet50", "vit_b_16", "convnext_tiny"]
    if xgb_specs:
        for name in xgb_specs:
            model_names.append(f"xgb_{name}")

    def get_all_model_probs(records, imgs, labels, split_name):
        print(f"\n[{split_name.upper()}] Running inference for {len(records)} samples...")
        pr_r, _ = infer(model_r, imgs, labels, device, tf_r, tta_n=tta_n, batch_size=args.batch_size)
        pr_r = align_probs_to_canonical(pr_r, lm_r["label_to_idx"], canonical_labels)
        
        pr_v, _ = infer(model_v, imgs, labels, device, tf_v, tta_n=tta_n, batch_size=args.batch_size)
        pr_v = align_probs_to_canonical(pr_v, lm_v["label_to_idx"], canonical_labels)
        
        pr_c, _ = infer(model_c, imgs, labels, device, tf_c, tta_n=tta_n, batch_size=args.batch_size)
        pr_c = align_probs_to_canonical(pr_c, lm_c["label_to_idx"], canonical_labels)

        probs = [pr_r, pr_v, pr_c]

        if xgb_specs:
            for name in xgb_specs:
                model_path, label_map_path, emb_npz = xgb_paths[name]
                pr_x = build_xgb_probs(
                    records=records,
                    metadata_json=Path(args.metadata_json),
                    embeddings_npz=emb_npz,
                    xgb_model_path=model_path,
                    xgb_label_map=label_map_path,
                    canonical_labels=canonical_labels,
                )
                probs.append(pr_x)
        return probs

    # Inference on val (for tuning)
    probs_val = get_all_model_probs(val_records, val_imgs, y_val, "val")

    print(f"\n--- Grid Search on VAL split (step={args.step}) ---")
    best_f1 = -1.0
    best_w = None
    
    # Calculate total combinations to give tqdm a total count
    import math
    n = round(1.0 / args.step)
    num_models = len(model_names)
    total_combinations = math.comb(n + num_models - 1, num_models - 1)
    
    try:
        from tqdm import tqdm
        sweep_iter = tqdm(simplex_grid(num_models=num_models, step=args.step), total=total_combinations, desc="Grid Search")
    except ImportError:
        sweep_iter = simplex_grid(num_models=num_models, step=args.step)
        
    for w in sweep_iter:
        ep = np.zeros_like(probs_val[0])
        for wi, pi in zip(w, probs_val):
            ep += wi * pi
        pred = ep.argmax(1)
        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_w = w

    w_str = " / ".join([f"{x:.2f}" for x in best_w])
    print(f"Best weights ({' / '.join(model_names)}): {w_str}")
    print(f"Best VAL Macro F1: {best_f1:.4f}")

    # Inference on Train and Test
    probs_train = get_all_model_probs(train_records, train_imgs, y_train, "train")
    probs_test = get_all_model_probs(test_records, test_imgs, y_test, "test")

    def eval_ensemble(probs, y_true, split_name):
        ep = np.zeros_like(probs[0])
        for wi, pi in zip(best_w, probs):
            ep += wi * pi
        pred = ep.argmax(1)
        f1 = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)
        cm = confusion_matrix(y_true, pred)
        print(f"\n[{split_name.upper()}] Final Ensemble Metrics:")
        print(f"  Macro F1 : {f1:.4f}")
        print(f"  Accuracy : {acc:.4f}")
        return f1, acc, cm

    f1_tr, acc_tr, cm_tr = eval_ensemble(probs_train, y_train, "train")
    f1_v, acc_v, cm_v = eval_ensemble(probs_val, y_val, "val")
    f1_te, acc_te, cm_te = eval_ensemble(probs_test, y_test, "test")

    print("\n" + "="*50)
    print("TRAIN Confusion Matrix:")
    print(cm_tr)
    print("\n" + "="*50)
    print("TEST Confusion Matrix:")
    print(cm_te)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
