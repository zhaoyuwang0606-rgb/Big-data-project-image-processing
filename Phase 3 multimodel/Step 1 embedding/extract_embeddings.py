"""
=============================================================================
File Name: extract_embeddings.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Extracts high-dimensional embeddings using pre-trained visual foundation models (CLIP, DINOv2, etc.).

Main Functionality/Workflow:
Passes all images through frozen foundation models and saves the resulting feature vectors to disk for late fusion.

Key Inputs Path:
Images directory, Split JSON files

Key Outputs Path:
.npz files containing extracted embeddings and labels

Important Dependencies:
torch, torchvision, transformers (if applicable), numpy

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 3 (Multimodal): Prepares deep semantic features for the XGBoost meta-classifier.
=============================================================================
"""

import argparse
import json
import os
import sys

# Force Transformers to use PyTorch and completely ignore the broken TensorFlow installation
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    pass # Will be handled gracefully inside the model builder if user chooses CLIP

def build_model_and_transforms(model_name, device):
    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        # Remove classification head to get 2048-d embeddings
        model.fc = nn.Identity()
        eval_t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ])
        
    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        # Remove classification head to get 768-d embeddings
        model.heads = nn.Identity()
        eval_t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ])
        
    elif model_name == "dinov2":
        # Load Meta's DINOv2 from Hugging Face instead of GitHub to avoid 403 Rate Limit
        if "CLIPModel" not in globals():
            print("\n[ERROR] The 'transformers' library is required for DINOv2. Please run: pip install transformers\n")
            sys.exit(1)
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
        def eval_t(img):
            inputs = processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)

    elif model_name == "clip":
        if "CLIPModel" not in globals():
            print("\n[ERROR] The 'transformers' library is not installed in your current environment.")
            print("Please run this command in your terminal first:")
            print("pip install transformers\n")
            sys.exit(1)
            
        # Load OpenAI's CLIP purely for its Vision Encoder
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        def eval_t(img):
            inputs = processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0) # remove batch dim
            
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()
    return model, eval_t

def extract_features_from_folder(image_paths, model, transform, device, model_name):
    embeddings = []
    filenames = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting", mininterval=2.0):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
                
            tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract Visual Embedding based on model type
            if model_name == "clip":
                emb = model(tensor).pooler_output.squeeze(0).cpu().numpy()
            elif model_name == "dinov2":
                emb = model(tensor).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            else:
                emb = model(tensor).squeeze(0).cpu().numpy()
                
            embeddings.append(emb)
            filenames.append(img_path.name)
            
    return np.array(embeddings), np.array(filenames)

def write_run_log(out_dir, args):
    log_file = Path(out_dir) / "run_log.json"
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "python " + " ".join(sys.argv),
        "args": vars(args),
        "cwd": os.getcwd()
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract purely visual embeddings directly from a folder of images.")
    parser.add_argument("--image-root", required=True, help="Path to images directory")
    parser.add_argument("--model", choices=["resnet50", "vit_b_16", "dinov2", "clip"], required=True)
    parser.add_argument("--out-root", default="Phase 3 multimodel/Step 1 embedding")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create separate output directory for each model
    out_dir = Path(args.out_root) / f"embeddings_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save execution log
    write_run_log(out_dir, args)

    print(f"Scanning images in {args.image_root}...")
    image_paths = sorted(list(Path(args.image_root).glob("*.jpg")))
    
    if len(image_paths) == 0:
        print(f"Error: No .jpg images found in {args.image_root}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images.")

    print(f"Loading {args.model} feature extractor...")
    model, transform = build_model_and_transforms(args.model, device)

    # Extract all features in one go
    emb, filenames = extract_features_from_folder(image_paths, model, transform, device, args.model)
    
    # Save the giant database of features
    out_file = out_dir / "all_features.npz"
    np.savez_compressed(out_file, embeddings=emb, filenames=filenames)
    print(f"\n[SUCCESS] Saved {emb.shape} embeddings to {out_file}")

if __name__ == "__main__":
    main()
