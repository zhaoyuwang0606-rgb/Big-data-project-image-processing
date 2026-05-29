"""
=============================================================================
File Name: train_xgboost.py
Author: Zhaoyu Wang
Date: 2026-05-29

Purpose:
Trains XGBoost classifiers on the extracted deep embeddings.

Main Functionality/Workflow:
Loads the .npz embeddings, performs hyperparameter tuning, trains XGBoost models, and evaluates classification performance.

Key Inputs Path:
Extracted .npz embedding files

Key Outputs Path:
Trained XGBoost models (.ubj), classification reports

Important Dependencies:
xgboost, scikit-learn, numpy

Reproducibility Notes:
Fixed seeds (e.g., seed=25) are utilized where applicable. Ensure the 
project root structure is maintained. Relative paths are resolved dynamically.

Pipeline Fit:
Phase 3 (Multimodal): Serves as the late-fusion classifier, combining advanced visual features to maximize accuracy.
=============================================================================
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import optuna
except ImportError:
    pass # Handled below

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_run_log(out_dir, args, output_files):
    from datetime import datetime, timezone
    log_file = Path(out_dir) / "run_log.json"
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "python " + " ".join(sys.argv),
        "args": vars(args),
        "cwd": os.getcwd(),
        "generated_outputs": [str(p) for p in output_files]
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def clean_price(price_str):
    if not price_str or price_str == "Not known":
        return np.nan
    # Extract digits and decimals, e.g. "~€4.21" -> 4.21
    match = re.search(r'([\d\.]+)', str(price_str))
    if match:
        return float(match.group(1))
    return np.nan

def clean_year(year_str):
    if not year_str or str(year_str) == "Not known":
        return np.nan
    try:
        return float(year_str)
    except:
        return np.nan

def build_metadata_dict(metadata_list):
    # Create a quick lookup dict from img_local_path -> metadata features
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

def prepare_dataset(split_records, emb_dict, meta_dict, label_to_idx):
    X_visual = []
    X_tabular = []
    y = []
    
    missing_features = 0
    
    for r in split_records:
        path = r.get("img_local_path")
        if not path:
            missing_features += 1
            continue
            
        filename = Path(path).name
        
        # Look up visual features
        if filename not in emb_dict:
            missing_features += 1
            continue
        v_feat = emb_dict[filename]
        
        # Look up tabular features (year, price_new, price_used)
        t_feat = meta_dict.get(path, [np.nan, np.nan, np.nan])
        
        # Look up label
        cat = r.get("category", "")
        if cat not in label_to_idx:
            continue
            
        X_visual.append(v_feat)
        X_tabular.append(t_feat)
        y.append(label_to_idx[cat])
        
    if missing_features > 0:
        print(f"  Warning: {missing_features} images in JSON were missing from .npz features.")
        
    # Concatenate visual and tabular
    X_v = np.array(X_visual, dtype=np.float32)
    X_t = np.array(X_tabular, dtype=np.float32)
    
    # Final Multi-Modal Vector
    X = np.hstack([X_v, X_t])
    y = np.array(y, dtype=np.int32)
    
    return X, y

def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    # Normalize by the true labels (rows sum to 1)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Since 123 classes is too huge for a single plot, let's plot a massive figure
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=classes, yticklabels=classes, fmt=".2f")
    plt.title("Normalized Multi-Modal Confusion Matrix (Recall per Class)")
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal XGBoost Classifier")
    parser.add_argument("--model", choices=["resnet50", "vit_b_16", "dinov2", "clip"], required=True)
    parser.add_argument("--embeddings-root", default="Phase 3 multimodel/Step 1 embedding")
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--train-json", default="Phase 2 splitting/split_metadata_threshold100_seed25_70_15_15_train.json")
    parser.add_argument("--val-json", default="Phase 2 splitting/split_metadata_threshold100_seed25_70_15_15_val.json")
    parser.add_argument("--test-json", default="Phase 2 splitting/split_metadata_threshold100_seed25_70_15_15_test.json")
    parser.add_argument("--out-dir", default="Phase 3 multimodel/Step 2 ML classifier/results")
    parser.add_argument("--top-k", type=int, default=0, help="If > 0, filter the dataset to only use the Top K most frequent categories (e.g., 30)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--load-params", default="", help="Path to best_hyperparameters.json to skip tuning and use saved params directly")
    args = parser.parse_args()

    # Setup output directory
    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {out_dir}")

    print("1. Loading raw metadata to extract Year and Prices...")
    full_metadata = load_json(args.metadata)
    meta_dict = build_metadata_dict(full_metadata)
    print(f"   Loaded metadata for {len(meta_dict)} items.")

    print(f"2. Loading visual features from {args.model} database...")
    npz_path = Path(args.embeddings_root) / f"embeddings_{args.model}" / "all_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run extract_embeddings.py first.")
    
    data = np.load(npz_path)
    emb_array = data['embeddings']
    fn_array = data['filenames']
    # Build a fast dictionary for filename -> embedding lookup
    emb_dict = {str(fn): emb for fn, emb in zip(fn_array, emb_array)}
    print(f"   Loaded {len(emb_dict)} visual embeddings (Dimension: {emb_array.shape[1]}).")

    print("3. Processing Splits and Multi-Modal Fusion...")
    train_recs = load_json(args.train_json)
    val_recs = load_json(args.val_json)
    test_recs = load_json(args.test_json)

    # Apply Top-K filtering if requested
    if args.top_k > 0:
        from collections import Counter
        print(f"   >>> Filtering dataset to only use the Top {args.top_k} most frequent categories...")
        # Find top k categories in train set
        counts = Counter([r.get("category", "") for r in train_recs if r.get("category", "")])
        top_cats = set([cat for cat, count in counts.most_common(args.top_k)])
        
        # Filter all splits
        train_recs = [r for r in train_recs if r.get("category", "") in top_cats]
        val_recs = [r for r in val_recs if r.get("category", "") in top_cats]
        test_recs = [r for r in test_recs if r.get("category", "") in top_cats]
        
        print(f"   >>> Filtered Train size: {len(train_recs)}, Val size: {len(val_recs)}, Test size: {len(test_recs)}")

    # Build Global Label Map from train
    unique_cats = sorted(list(set(r.get("category", "") for r in train_recs)))
    label_to_idx = {c: i for i, c in enumerate(unique_cats)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    # Save Label Map
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    X_train, y_train = prepare_dataset(train_recs, emb_dict, meta_dict, label_to_idx)
    X_val, y_val = prepare_dataset(val_recs, emb_dict, meta_dict, label_to_idx)
    X_test, y_test = prepare_dataset(test_recs, emb_dict, meta_dict, label_to_idx)

    print(f"   X_train shape: {X_train.shape} (Visual + 3 Tabular Meta)")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   X_test shape:  {X_test.shape}")

    best_params = {
        "n_estimators": 1000,
        "tree_method": "hist",
        "device": "cuda",
        "early_stopping_rounds": 20,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1
    }

    if args.tune:
        if "optuna" not in globals():
            print("\n[ERROR] The 'optuna' library is required for tuning. Please run: pip install optuna")
            sys.exit(1)
            
        print(f"\n[START] Starting Optuna Hyperparameter Tuning ({args.n_trials} trials)...")
        
        def objective(trial):
            param = {
                "n_estimators": 1000,
                "tree_method": "hist",
                "device": "cuda",
                "early_stopping_rounds": 20,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            
            model = xgb.XGBClassifier(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50  # Let it print every 50 trees so you can see the progress!
            )
            
            y_val_pred = model.predict(X_val)
            # Optimize for Validation Macro F1
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            return val_f1
            
        # We want to MAXIMIZE the Macro F1 score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        
        print("\n[DONE] Tuning Completed!")
        print(f"Best Validation Macro F1: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
            
        # Update our best_params with the optimized ones
        best_params.update(study.best_params)
        
        # Save the best params to disk
        with open(out_dir / "best_hyperparameters.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

    # Load pre-saved hyperparameters if --load-params is specified
    if args.load_params and Path(args.load_params).exists():
        print(f"\n[INFO] Loading hyperparameters from {args.load_params}...")
        with open(args.load_params, "r") as f:
            loaded_params = json.load(f)
        best_params.update(loaded_params)
        print(f"  Loaded params: {loaded_params}")

    print("\n4. Training Final XGBoost Classifier...")
    clf = xgb.XGBClassifier(**best_params)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10 if not args.tune else False # less verbose if tuned
    )

    # Save the trained model so it can be loaded later for ensemble inference
    model_save_path = out_dir / "xgb_model.ubj"
    clf.save_model(model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

    print("\n5. Evaluating on Test Set...")
    # Predict classes
    y_pred = clf.predict(X_test)
    # Predict probabilities (for AUC)
    y_prob = clf.predict_proba(X_test)

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    try:
        # Multi-class AUC can be tricky if a class is missing in test set
        auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro", labels=list(range(len(unique_cats))))
    except ValueError:
        auc = float('nan')
        print("   Warning: Could not compute AUC (likely some classes missing in test set).")

    print("-" * 30)
    print("FINAL METRICS (Test Set)")
    print("-" * 30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print(f"Macro AUC: {auc:.4f}")
    print("-" * 30)

    # Save metrics to JSON
    metrics = {
        "model": args.model,
        "features_dim": X_train.shape[1],
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1),
        "test_macro_auc": float(auc)
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Generating Confusion Matrix (saving to disk due to large size)...")
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, [idx_to_label[i] for i in range(len(unique_cats))], cm_path)
    
    # Save running log for academic tracking
    output_files = [out_dir / "label_map.json", out_dir / "test_metrics.json", cm_path, model_save_path]
    if args.tune:
        output_files.append(out_dir / "best_hyperparameters.json")
    write_run_log(out_dir, args, output_files)
    
    print(f"\n[SUCCESS] Pipeline finished. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
