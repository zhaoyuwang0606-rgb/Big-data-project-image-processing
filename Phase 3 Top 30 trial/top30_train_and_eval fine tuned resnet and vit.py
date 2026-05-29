"""
=============================================================================
File Name: top30_train_and_eval fine tuned resnet and vit.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Trains and evaluates ResNet-50 and ViT-B/16 on the Top 30 LEGO categories.

Main Functionality/Workflow:
Loads data using PyTorch Datasets, applies augmentations, fine-tunes pretrained models, tracks loss/accuracy, and outputs evaluation metrics.

Key Inputs Path:
Images directory, Split JSON files

Key Outputs Path:
Trained model weights (.pt), loss curves, confusion matrices

Important Dependencies:
torch, torchvision, sklearn, matplotlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 3 (CV Baseline): Establishes baseline performance using standard CNN and Transformer architectures.
=============================================================================
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16


# ==========================================
# [Utility Module 1] Path Processing & Command Reproduction
# Provides project path resolution to save the exact execution command for future reproducibility.
# ==========================================
def project_root():
    return Path(__file__).resolve().parents[1]


def relpath(path, base_dir):
    return os.path.relpath(str(Path(path).resolve()), start=str(Path(base_dir).resolve()))


def build_repro_command(args):
    root = project_root()
    cmd = [
        "python",
        relpath(Path(__file__).resolve(), root),
        "--train-json",
        relpath(args.train_json, root),
        "--val-json",
        relpath(args.val_json, root),
        "--test-json",
        relpath(args.test_json, root),
        "--image-root",
        relpath(args.image_root, root),
        "--out-dir",
        relpath(args.out_dir, root),
        "--model",
        args.model,
        "--label-field",
        args.label_field,
        "--epochs",
        str(args.epochs),
        "--freeze-epochs",
        str(args.freeze_epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.max_steps_per_epoch is not None:
        cmd += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
    if args.device:
        cmd += ["--device", args.device]
    return subprocess_list2cmdline(cmd)


def subprocess_list2cmdline(cmd):
    import subprocess

    return subprocess.list2cmdline(cmd)


def write_run_log(out_dir, outputs, command):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    root = project_root()

    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== {ts} ===\n")
        f.write("cwd: .\n")
        f.write(f"command: {command}\n")
        f.write("outputs:\n")
        for p in outputs:
            f.write(f"- {relpath(p, root)}\n")
        f.write("\n")


# ==========================================
# [Utility Module 2] Random Seed & Reproducibility
# Fixes random number generators (Python, Numpy, PyTorch) to ensure consistent results across runs.
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(img_local_path, image_root, minifig_number=None):
    image_root = Path(image_root)
    name = Path(img_local_path).name if img_local_path else ""
    stem = Path(name).stem if name else ""
    candidates = []
    if minifig_number:
        ms = str(minifig_number)
        candidates.extend(
            [
                image_root / f"{ms}.jpg",
                image_root / f"{ms}.png",
                image_root / f"{ms.lower()}.jpg",
                image_root / f"{ms.lower()}.png",
                image_root / f"{ms.upper()}.jpg",
                image_root / f"{ms.upper()}.png",
            ]
        )
    if name:
        candidates.extend(
            [
                image_root / name,
                image_root / f"{stem}.jpg",
                image_root / f"{stem}.png",
                image_root / f"{stem.lower()}.jpg",
                image_root / f"{stem.lower()}.png",
                image_root / f"{stem.upper()}.jpg",
                image_root / f"{stem.upper()}.png",
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    return None


# ==========================================
# [Data Loading Module] PyTorch Dataset Class (MinifigDataset)
# Responsible for reading Lego minifigure metadata, resolving image paths, loading images, and applying Transforms.
# ==========================================
class MinifigDataset(Dataset):
    def __init__(self, records, image_root, label_to_idx, label_field, transform):
        self.records = records
        self.image_root = Path(image_root)
        self.label_to_idx = label_to_idx
        self.label_field = label_field
        self.transform = transform

        missing = 0
        for r in self.records:
            if resolve_image_path(r.get("img_local_path"), self.image_root, r.get("minifig_number")) is None:
                missing += 1
        if missing:
            raise FileNotFoundError(f"Missing {missing}/{len(self.records)} images under {self.image_root}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_path = resolve_image_path(r.get("img_local_path"), self.image_root, r.get("minifig_number"))
        if img_path is None:
            raise FileNotFoundError(r.get("img_local_path"))
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = self.label_to_idx[r[self.label_field]]
        return x, y, r.get("minifig_number", ""), r.get(self.label_field, "")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ==========================================
# [Label Mapping Module] Target Category to Index Mapping
# Automatically extracts all categories from the training set to create string label to integer index mappings (Label Encoding).
# ==========================================
def build_label_map(train_records, label_field):
    labels = sorted({r[label_field] for r in train_records})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    return labels, label_to_idx, idx_to_label


# ==========================================
# [Data Augmentation & Preprocessing Module] Transforms for Train/Val Sets
# Contains advanced augmentation strategies designed for long-tail data (e.g., Random Erasing, Color Jitter) to improve generalization.
# ==========================================
def build_transforms(model_name, random_erasing_prob=0.0):
    """
    Builds image preprocessing and data augmentation pipelines.
    Crucial for improving model generalization, especially for long-tail, imbalanced datasets like Lego minifigures.
    """
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        mean, std = weights.transforms().mean, weights.transforms().std
        size = 224
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
        mean, std = weights.transforms().mean, weights.transforms().std
        size = 224

    train_t = transforms.Compose(
        [
            # 1. Random Resized Crop: Forces the model to focus on local features (e.g., torso prints) instead of just the global shape.
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            # 2. Random Horizontal Flip: Increases data diversity.
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. Random Rotation: Handles Lego minifigures photographed from various tilted angles.
            transforms.RandomRotation(degrees=15),
            # 4. Color Jitter: Alters brightness, contrast, saturation to prevent overfitting to specific lighting conditions.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            # 5. Normalization: Uses ImageNet pre-trained mean and std to accelerate convergence.
            transforms.Normalize(mean=mean, std=std),
            # 6. Random Erasing: Randomly occludes parts of the image, forcing the model to learn alternative distinguishing features (robustness against occlusion).
            transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ]
    )
    eval_t = transforms.Compose(
        [
            # Only basic resizing and center cropping for validation/test sets to ensure fair evaluation.
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_t, eval_t


# ==========================================
# [CutMix Advanced Augmentation Module] Random Bounding Box Generator
# Generates random rectangular region coordinates within an image based on a given lambda ratio, used for mixing two images.
# ==========================================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# ==========================================
# [Model Construction Module] Initialize Pre-trained Models & Replace Heads
# Supports loading pre-trained ResNet50 or ViT-B/16, automatically adding heavy Dropout (p=0.5) to prevent overfitting.
# ==========================================
def build_model(model_name, num_classes):
    """
    Builds a pre-trained model (ResNet50 or ViT-B/16) and replaces the final classification head.
    """
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        # Heavy Regularization: Dropout (p=0.5) before the final linear layer prevents co-adaptation and overfitting.
        model.heads.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
        head_names = {"heads.head"}
        return model, head_names

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    # Heavy Regularization: Dropout (p=0.5) before the final linear layer prevents co-adaptation and overfitting.
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    head_names = {"fc"}
    return model, head_names


# ==========================================
# [Model Fine-Tuning Control Module] Gradient Switches (Two-Stage Fine-tuning)
# Determines whether to freeze the backbone (training only the head) or unfreeze the entire network for full-parameter fine-tuning.
# ==========================================
def set_trainable(model, head_names, train_backbone):
    for name, p in model.named_parameters():
        if any(name == hn or name.startswith(hn + ".") for hn in head_names):
            p.requires_grad = True
        else:
            p.requires_grad = bool(train_backbone)


@dataclass
class EvalOutput:
    loss: float
    acc: float
    macro_f1: float
    auc_ovr_macro: Optional[float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    minifig_numbers: list
    true_labels: list
    pred_labels: list


# ==========================================
# [Model Evaluation Module] Core Evaluation Metrics Calculation
# Returns detailed evaluation results including Loss, Accuracy, Macro-F1 (used for smoothing), and Multi-class AUC.
# ==========================================
def evaluate(model, loader, device, criterion, idx_to_label):
    model.eval()
    losses = []
    all_y = []
    all_pred = []
    all_prob = []
    all_ids = []
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for x, y, minifig_number, true_label_str in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_ids.extend(list(minifig_number))
            all_true_labels.extend(list(true_label_str))
            all_pred_labels.extend([idx_to_label[int(i)] for i in pred.cpu().numpy().tolist()])

    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob) if all_prob else np.zeros((0, len(idx_to_label)), dtype=np.float32)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    auc = None
    if len(y_true):
        try:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = None

    return EvalOutput(
        loss=float(np.mean(losses)) if losses else 0.0,
        acc=acc,
        macro_f1=macro_f1,
        auc_ovr_macro=auc,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        minifig_numbers=all_ids,
        true_labels=all_true_labels,
        pred_labels=all_pred_labels,
    )


# ==========================================
# [Result Visualization Module 1] Plot & Save Confusion Matrix
# Intuitively displays the model's prediction accuracy across categories and highlights easily confused classes (supports normalization).
# ==========================================
def plot_confusion_matrix(cm, labels, out_path, normalize):
    import matplotlib.pyplot as plt

    cm = cm.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ==========================================
# [Result Visualization Module 2] Plot & Save Learning Curves
# Generates training and validation Loss and Accuracy trend graphs based on the training history.
# ==========================================
def plot_loss_curves(metrics_csv, out_dir):
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(metrics_csv).open("r", encoding="utf-8", newline="")))
    train_epochs, train_loss = [], []
    val_epochs, val_loss = [], []
    for r in rows:
        if not r.get("epoch") or not r.get("loss") or not r.get("split"):
            continue
        e = int(r["epoch"])
        l = float(r["loss"])
        if r["split"] == "train":
            train_epochs.append(e)
            train_loss.append(l)
        elif r["split"] == "val":
            val_epochs.append(e)
            val_loss.append(l)

    out_dir = Path(out_dir)
    out_png = out_dir / "loss_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes[0], axes[1]

    if train_epochs:
        ax0.plot(train_epochs, train_loss, marker="o")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("Train Loss")

    if val_epochs:
        ax1.plot(val_epochs, val_loss, marker="o", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Val Loss")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


# ==========================================
# [Feature Interpretability Module (Grad-CAM)] Gradient-based Class Activation Maps
# Includes specific image preprocessing, gradient capture hooks, and heatmap overlay function definitions.
# ==========================================
def build_eval_crop_transform(model_name):
    if model_name == "vit_b_16":
        size = 224
    else:
        size = 224
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size)])


def build_normalize_transform(model_name):
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
    mean, std = weights.transforms().mean, weights.transforms().std
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def to_uint8_image(pil_img):
    arr = np.array(pil_img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def overlay_heatmap(rgb_uint8, heatmap_0_1, alpha=0.45):
    import matplotlib.cm as cm

    heatmap_0_1 = np.clip(heatmap_0_1, 0.0, 1.0)
    colored = cm.get_cmap("jet")(heatmap_0_1)[:, :, :3]
    colored_uint8 = (colored * 255.0).astype(np.uint8)
    out = (1.0 - alpha) * rgb_uint8.astype(np.float32) + alpha * colored_uint8.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def gradcam_for_model(model, input_tensor, class_idx, model_name):
    if model_name == "resnet50":
        target_layer = model.layer4[-1]
    elif model_name == "vit_b_16":
        target_layer = model.encoder.layers[-1]
    else:
        raise ValueError(f"GradCAM not implemented for {model_name}")

    activations = None
    gradients = None

    def fwd_hook(_, __, out):
        nonlocal activations
        activations = out

    def bwd_hook(_, __, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(input_tensor)
        score = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if activations is None or gradients is None:
            raise RuntimeError("GradCAM hooks did not capture tensors")
            
        if model_name == "vit_b_16":
            act = activations[:, 1:, :]
            grad = gradients[:, 1:, :]
            hw = int(np.sqrt(act.shape[1]))
            act = act.transpose(1, 2).reshape(1, -1, hw, hw)
            grad = grad.transpose(1, 2).reshape(1, -1, hw, hw)
        else:
            act = activations
            grad = gradients

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def write_gradcam_gallery(model, records_by_id, image_root, out_dir, model_name, eval_out, max_images=16, seed=25):
    if model_name not in ["resnet50", "vit_b_16"]:
        return None

    rng = random.Random(seed)
    ids = eval_out.minifig_numbers
    y_true = eval_out.y_true
    y_pred = eval_out.y_pred
    y_prob = eval_out.y_prob

    correct = [ids[i] for i in range(len(ids)) if int(y_true[i]) == int(y_pred[i])]
    wrong = [ids[i] for i in range(len(ids)) if int(y_true[i]) != int(y_pred[i])]
    rng.shuffle(correct)
    rng.shuffle(wrong)

    n_wrong = min(len(wrong), max_images // 2)
    n_correct = min(len(correct), max_images - n_wrong)
    selected = wrong[:n_wrong] + correct[:n_correct]

    crop_t = build_eval_crop_transform(model_name)
    norm_t = build_normalize_transform(model_name)

    heatmap_dir = Path(out_dir) / "heatmaps_test_gradcam"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = heatmap_dir / "heatmaps_manifest.csv"

    rows = []
    device = next(model.parameters()).device
    model.eval()

    for mid in selected:
        rec = records_by_id.get(mid)
        if rec is None:
            continue
        img_path = resolve_image_path(rec.get("img_local_path"), image_root, rec.get("minifig_number"))
        if img_path is None:
            continue

        pil = Image.open(img_path).convert("RGB")
        pil_crop = crop_t(pil)
        rgb_uint8 = to_uint8_image(pil_crop)

        x = norm_t(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

        true_label = rec.get("category_grouped_threshold_100", rec.get("category", ""))
        true_idx = None
        for i in range(len(y_true)):
            if ids[i] == mid:
                true_idx = int(y_true[i])
                break
        if true_idx is None:
            true_idx = pred_idx

        cam = gradcam_for_model(model, x, class_idx=pred_idx, model_name=model_name)
        overlay = overlay_heatmap(rgb_uint8, cam, alpha=0.45)

        out_name = f"{mid}__true_{true_idx}__pred_{pred_idx}__p_{pred_conf:.3f}.png"
        out_path = heatmap_dir / out_name
        Image.fromarray(overlay).save(out_path)

        rows.append([mid, str(relpath(img_path, project_root())), int(true_idx), int(pred_idx), pred_conf, str(relpath(out_path, project_root()))])

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minifig_number", "image_path", "true_idx", "pred_idx", "pred_conf", "heatmap_path"])
        w.writerows(rows)

    return {"dir": heatmap_dir, "manifest": manifest_path}


def write_predictions_csv(path, split_name, eval_out):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "minifig_number", "true_label", "pred_label"])
        for mid, t, p in zip(eval_out.minifig_numbers, eval_out.true_labels, eval_out.pred_labels):
            w.writerow([split_name, mid, t, p])


# ==========================================
# [Main Workflow Control Module] Entry Function main()
# Responsible for parsing hyperparameters, loading/splitting data, executing the full training loop (including CutMix, SMA early stopping), final evaluation, and reporting.
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--model", choices=["resnet50", "vit_b_16"], default="resnet50")
    parser.add_argument("--label-field", default="category_grouped_threshold_100")
    parser.add_argument("--top-k", type=int, default=None, help="If set, only train on the top K most frequent categories.")
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-interval-steps", type=int, default=50)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from (e.g. last.pt)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = args.device.strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    command = build_repro_command(args)
    write_run_log(out_dir=out_dir, outputs=[], command=command)
    progress_path = out_dir / "progress.jsonl"
    progress_f = progress_path.open("a", encoding="utf-8")

    def log_event(obj):
        line = json.dumps(obj, ensure_ascii=False)
        print(line)
        progress_f.write(line + "\n")
        progress_f.flush()

    train_records = load_json(args.train_json)
    val_records = load_json(args.val_json)
    test_records = load_json(args.test_json)

    if args.top_k is not None:
        import collections
        print(f"Filtering dataset to Top {args.top_k} categories based on original 'category' field...")
        category_counts = collections.Counter(r.get("category", "") for r in train_records)
        top_categories = {cat for cat, _ in category_counts.most_common(args.top_k)}
        
        train_records = [r for r in train_records if r.get("category") in top_categories]
        val_records = [r for r in val_records if r.get("category") in top_categories]
        test_records = [r for r in test_records if r.get("category") in top_categories]
        
        args.label_field = "category"
        print(f"Kept {len(train_records)} train, {len(val_records)} val, {len(test_records)} test images across {args.top_k} classes.")

    labels, label_to_idx, idx_to_label = build_label_map(train_records, args.label_field)
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": labels, "label_to_idx": label_to_idx}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_t, eval_t = build_transforms(args.model, args.random_erasing_prob)
    train_ds = MinifigDataset(train_records, args.image_root, label_to_idx, args.label_field, train_t)
    val_ds = MinifigDataset(val_records, args.image_root, label_to_idx, args.label_field, eval_t)
    test_ds = MinifigDataset(test_records, args.image_root, label_to_idx, args.label_field, eval_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model, head_names = build_model(args.model, num_classes=len(labels))
    model = model.to(device)

    # Label Smoothing (0.1): Prevents the model from becoming overconfident.
    # Highly effective for long-tail distributions and visually ambiguous categories.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    optimizer = None
    scheduler = None
    current_train_backbone = None

    metrics_csv = out_dir / "metrics_history.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    best_val_f1 = -1.0
    best_val_sma = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    start_epoch = 1
    val_f1_history = []

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        
        if metrics_csv.exists():
            with metrics_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split") == "val" and row.get("macro_f1"):
                        try:
                            val = float(row["macro_f1"])
                            if val > best_val_f1:
                                best_val_f1 = val
                        except ValueError:
                            pass
            print(f"Recovered best_val_f1: {best_val_f1}")
    else:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "split", "loss", "acc", "macro_f1", "auc_ovr_macro"])

    training_start = time.perf_counter()
    avg_epoch_seconds = None

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_backbone = epoch > args.freeze_epochs
            if train_backbone != current_train_backbone:
                set_trainable(model, head_names=head_names, train_backbone=train_backbone)
                # Optimizer: AdamW (Adam with Weight Decay) for better generalization.
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
                # Scheduler: Cosine Annealing smoothly decays the learning rate following a cosine curve.
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - epoch + 1, 1))
                current_train_backbone = train_backbone

            model.train()
            train_losses = []
            train_correct = 0
            train_seen = 0
            total_steps = args.max_steps_per_epoch if args.max_steps_per_epoch is not None else len(train_loader)
            print_every = max(1, min(int(args.log_interval_steps), int(total_steps) if total_steps else 1))
            avg_step_seconds = None
            last_step_t = time.perf_counter()
            step = 0
            log_event(
                {
                    "epoch": epoch,
                    "epochs_total": args.epochs,
                    "train_backbone": bool(train_backbone),
                    "steps_total": int(total_steps),
                    "device": device,
                }
            )
            for x, y, _, _ in train_loader:
                step += 1
                if args.max_steps_per_epoch is not None and step > args.max_steps_per_epoch:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    # CutMix Data Augmentation: Replaces a random rectangular patch of the image
                    # with a patch from another image, and mixes their labels proportionally.
                    # Forces the model to recognize partial objects.
                    if args.cutmix_prob > 0 and np.random.rand() < args.cutmix_prob:
                        lam = np.random.beta(1.0, 1.0)
                        rand_index = torch.randperm(x.size()[0]).to(device)
                        target_a = y
                        target_b = y[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                        
                        logits = model(x)
                        loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())
                pred = torch.argmax(logits, dim=1)
                train_correct += int((pred == y).sum().item())
                train_seen += int(y.numel())

                now = time.perf_counter()
                dt = now - last_step_t
                last_step_t = now
                if avg_step_seconds is None:
                    avg_step_seconds = dt
                else:
                    avg_step_seconds = 0.9 * avg_step_seconds + 0.1 * dt

                if avg_step_seconds and (step % print_every == 0 or step == 1 or step == total_steps):
                    done = step
                    remaining = max(total_steps - done, 0)
                    eta_epoch_s = remaining * avg_step_seconds
                    elapsed_epoch_s = now - epoch_start
                    msg = {
                        "epoch": epoch,
                        "step": done,
                        "steps_total": int(total_steps),
                        "loss_batch": float(loss.item()),
                        "loss_running_mean": float(np.mean(train_losses)) if train_losses else None,
                        "sec_per_step_ema": float(avg_step_seconds),
                        "elapsed_epoch_sec": float(elapsed_epoch_s),
                        "eta_epoch_sec": float(eta_epoch_s),
                    }
                    if avg_epoch_seconds is not None:
                        remaining_epochs = max(args.epochs - epoch, 0)
                        msg["eta_total_sec"] = float(remaining_epochs * avg_epoch_seconds + eta_epoch_s)
                    log_event(msg)

            scheduler.step()

            eval_start = time.perf_counter()
            val_out = evaluate(model, val_loader, device, criterion, idx_to_label)
            eval_seconds = time.perf_counter() - eval_start

            train_loss_epoch = float(np.mean(train_losses)) if train_losses else 0.0
            train_acc_epoch = float(train_correct / train_seen) if train_seen else 0.0

            with metrics_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([epoch, "train", train_loss_epoch, train_acc_epoch, "", ""])
                w.writerow([epoch, "val", val_out.loss, val_out.acc, val_out.macro_f1, val_out.auc_ovr_macro])

            epoch_seconds = time.perf_counter() - epoch_start
            if avg_epoch_seconds is None:
                avg_epoch_seconds = epoch_seconds
            else:
                avg_epoch_seconds = 0.7 * avg_epoch_seconds + 0.3 * epoch_seconds

            log_event(
                {
                    "epoch": epoch,
                    "epoch_sec": float(epoch_seconds),
                    "eval_sec": float(eval_seconds),
                    "train": {"loss": train_loss_epoch, "acc": train_acc_epoch},
                    "val": {"loss": val_out.loss, "acc": val_out.acc, "macro_f1": val_out.macro_f1, "auc_ovr_macro": val_out.auc_ovr_macro},
                    "elapsed_total_sec": float(time.perf_counter() - training_start),
                    "eta_total_sec": float(max(args.epochs - epoch, 0) * avg_epoch_seconds),
                }
            )

            torch.save({"model": model.state_dict(), "labels": labels, "args": vars(args), "epoch": epoch}, last_path)
            val_f1_history.append(val_out.macro_f1)
            if len(val_f1_history) > 3:
                val_f1_history.pop(0)
            # Calculate 3-Epoch Simple Moving Average (SMA) of Macro-F1.
            # Smooths out validation metric spikes/drops typical in long-tail datasets,
            # ensuring early stopping is based on a stable trend.
            current_sma = sum(val_f1_history) / len(val_f1_history)

            # Raw F1 tracker (only for saving the peak weights)
            if val_out.macro_f1 > best_val_f1:
                best_val_f1 = val_out.macro_f1
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "labels": labels, "args": vars(args), "epoch": epoch}, best_path)

            # SMA tracker (for early stopping trend control)
            improved_sma = current_sma > (best_val_sma + args.early_stop_min_delta)
            if improved_sma:
                best_val_sma = current_sma
                no_improve_epochs = 0
            else:
                if args.early_stop_patience > 0 and epoch >= args.early_stop_start_epoch:
                    no_improve_epochs += 1
                    if no_improve_epochs >= args.early_stop_patience:
                        log_event(
                            {
                                "early_stop": True,
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "best_val_macro_f1": best_val_f1,
                                "best_val_sma": best_val_sma,
                                "patience": args.early_stop_patience,
                                "min_delta": args.early_stop_min_delta,
                                "no_improve_epochs": no_improve_epochs,
                            }
                        )
                        break
    finally:
        progress_f.close()

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    train_out = evaluate(model, train_eval_loader, device, criterion, idx_to_label)
    test_out = evaluate(model, test_loader, device, criterion, idx_to_label)

    train_cm = confusion_matrix(train_out.y_true, train_out.y_pred, labels=list(range(len(labels))))
    test_cm = confusion_matrix(test_out.y_true, test_out.y_pred, labels=list(range(len(labels))))

    cm_train_png = out_dir / "confusion_matrix_train.png"
    cm_train_norm_png = out_dir / "confusion_matrix_train_normalized.png"
    cm_test_png = out_dir / "confusion_matrix_test.png"
    cm_test_norm_png = out_dir / "confusion_matrix_test_normalized.png"
    plot_confusion_matrix(train_cm, labels, cm_train_png, normalize=False)
    plot_confusion_matrix(train_cm, labels, cm_train_norm_png, normalize=True)
    plot_confusion_matrix(test_cm, labels, cm_test_png, normalize=False)
    plot_confusion_matrix(test_cm, labels, cm_test_norm_png, normalize=True)

    pred_train_csv = out_dir / "predictions_train.csv"
    pred_test_csv = out_dir / "predictions_test.csv"
    write_predictions_csv(pred_train_csv, "train", train_out)
    write_predictions_csv(pred_test_csv, "test", test_out)

    loss_curves_png = plot_loss_curves(metrics_csv, out_dir)

    test_records_by_id = {r.get("minifig_number"): r for r in test_records if r.get("minifig_number")}
    heatmap_info = write_gradcam_gallery(
        model=model,
        records_by_id=test_records_by_id,
        image_root=args.image_root,
        out_dir=out_dir,
        model_name=args.model,
        eval_out=test_out,
        max_images=16,
        seed=args.seed,
    )

    summary = {
        "model": args.model,
        "label_field": args.label_field,
        "num_classes": len(labels),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "train": {"loss": train_out.loss, "acc": train_out.acc, "macro_f1": train_out.macro_f1, "auc_ovr_macro": train_out.auc_ovr_macro},
        "test": {"loss": test_out.loss, "acc": test_out.acc, "macro_f1": test_out.macro_f1, "auc_ovr_macro": test_out.auc_ovr_macro},
        "paths": {
            "best_ckpt": str(best_path),
            "last_ckpt": str(last_path),
            "metrics_history": str(metrics_csv),
            "label_map": str(out_dir / "label_map.json"),
            "cm_train": str(cm_train_png),
            "cm_train_norm": str(cm_train_norm_png),
            "cm_test": str(cm_test_png),
            "cm_test_norm": str(cm_test_norm_png),
            "pred_train": str(pred_train_csv),
            "pred_test": str(pred_test_csv),
            "loss_curves": str(loss_curves_png),
        },
    }
    if heatmap_info is not None:
        summary["paths"]["heatmaps_dir"] = str(heatmap_info["dir"])
        summary["paths"]["heatmaps_manifest"] = str(heatmap_info["manifest"])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = [
        out_dir / "run.log",
        out_dir / "label_map.json",
        metrics_csv,
        best_path,
        last_path,
        out_dir / "summary.json",
        cm_train_png,
        cm_train_norm_png,
        cm_test_png,
        cm_test_norm_png,
        pred_train_csv,
        pred_test_csv,
        loss_curves_png,
    ]
    if heatmap_info is not None:
        outputs.append(heatmap_info["manifest"])
        outputs.append(heatmap_info["dir"])
    write_run_log(out_dir=out_dir, outputs=outputs, command=command)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()