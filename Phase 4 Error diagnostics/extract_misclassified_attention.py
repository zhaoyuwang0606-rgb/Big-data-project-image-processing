"""
=============================================================================
File Name: extract_misclassified_attention.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Generates Grad-CAM heatmaps for misclassified images to diagnose model attention.

Main Functionality/Workflow:
Filters test predictions for errors, runs images through ConvNeXt with backward hooks, and overlays attention heatmaps on original images.

Key Inputs Path:
Images directory, ConvNeXt weights, predictions_test.csv

Key Outputs Path:
Misclassified Grad-CAM heatmaps (.png)

Important Dependencies:
torch, torchvision, PIL, numpy, matplotlib

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 4 (Diagnostics): Visually demonstrates that the model extracts logical features despite incorrect commercial labels.
=============================================================================
"""

import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

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

def gradcam_for_model(model, input_tensor, class_idx, model_name="convnext_tiny"):
    target_layer = model.features[-1][-1]  # last CNBlock of last stage
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

def resolve_image_path(img_root, mf_num):
    img_root = Path(img_root)
    ms = str(mf_num)
    candidates = [
        img_root / f"{ms}.jpg",
        img_root / f"{ms}.png",
        img_root / f"{ms.lower()}.jpg",
        img_root / f"{ms.lower()}.png",
        img_root / f"{ms.upper()}.jpg",
        img_root / f"{ms.upper()}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def main():
    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "Phase 3 Top 30 trial" / "top30_convnext_tiny" / "result convnext"
    out_dir = base_dir / "Phase 4 Error diagnostics" / "misclassified_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load label map
    with open(model_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_info = json.load(f)
    labels = label_info["labels"]
    label_to_idx = label_info["label_to_idx"]

    # 2. Build model and load weights
    num_classes = len(labels)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier = torch.nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features, num_classes),
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    ckpt_path = model_dir.parent / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Load metadata and test predictions
    with open(base_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mf_dict = {str(item.get("minifig_number")): item for item in metadata}

    test_csv = model_dir / "predictions_test.csv"
    misclassified = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test" and row["true_label"] != row["pred_label"]:
                misclassified.append(row)

    # Transforms
    size = 224
    mean, std = weights.transforms().mean, weights.transforms().std
    # Using CenterCrop to reflect EXACTLY what the model saw during evaluation.
    # The image is cropped because that is how the model processes images in Phase 3.
    # To prevent cropping, we can use transforms.Resize((224, 224)) directly.
    crop_t = transforms.Compose([transforms.Resize((224, 224))])
    norm_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    output_data = []
    
    print(f"Found {len(misclassified)} misclassified images. Generating Grad-CAM and gathering metadata...")

    for i, row in enumerate(misclassified):
        mf_num = row["minifig_number"]
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        
        img_path = resolve_image_path(base_dir / "images" / "images", mf_num)
        if not img_path:
            print(f"Warning: Image not found for {mf_num}")
            continue

        # Load image
        pil = Image.open(img_path).convert("RGB")
        pil_crop = crop_t(pil)
        rgb_uint8 = to_uint8_image(pil_crop)
        x = norm_t(pil_crop).unsqueeze(0).to(device)

        # Inference to get confidence
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            
        pred_idx = label_to_idx[pred_label]
        pred_conf = float(probs[pred_idx])

        # Grad-CAM on predicted label (to see why it predicted wrong)
        cam = gradcam_for_model(model, x, class_idx=pred_idx, model_name="convnext_tiny")
        overlay = overlay_heatmap(rgb_uint8, cam, alpha=0.45)
        
        # Safe filename
        safe_true = true_label.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_pred = pred_label.replace("/", "_").replace("\\", "_").replace(":", "_")
        out_name = f"{mf_num}_T-{safe_true}_P-{safe_pred}_conf{pred_conf:.2f}.png"
        
        Image.fromarray(overlay).save(out_dir / out_name)

        # Gather metadata
        meta = mf_dict.get(mf_num, {})
        entry = {
            "minifig_number": mf_num,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": pred_conf,
            "name": meta.get("name", ""),
            "subcategory": meta.get("subcategory", ""),
            "themes": meta.get("themes", []),
            "heatmap_file": out_name
        }
        output_data.append(entry)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(misclassified)}")

    # Save to JSON
    with open(out_dir / "misclassified_metadata.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Save to CSV
    csv_path = out_dir / "misclassified_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["minifig_number", "true_label", "pred_label", "confidence", "name", "subcategory", "themes", "heatmap_file"])
        writer.writeheader()
        for row in output_data:
            # Join themes list to string for CSV
            row_copy = row.copy()
            if isinstance(row_copy["themes"], list):
                row_copy["themes"] = " | ".join(row_copy["themes"])
            writer.writerow(row_copy)

    print(f"\nDone! Generated {len(output_data)} heatmaps.")
    print(f"Metadata saved to: {out_dir / 'misclassified_metadata.json'}")
    print(f"Heatmaps saved in: {out_dir}")

if __name__ == "__main__":
    main()