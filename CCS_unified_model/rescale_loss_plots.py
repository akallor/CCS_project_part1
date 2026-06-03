#!/usr/bin/env python3
"""
Rescale training loss from normalized scale to original CCS scale and regenerate plots.
====================================================================================

Reads training_history.json and (optionally) the same data used for training to get
target_std. Converts MSE (normalized) to RMSE in original CCS units: RMSE_original = target_std * sqrt(MSE_normalized).
R² values are unchanged (scale-invariant). Saves the same style of plots (Loss, R²) in PNG, PDF, and SVG.

Usage:
  python rescale_loss_plots.py --history_path unified_esm_results_corrected/training_history.json --output_dir unified_esm_results_corrected_rescaled --data_path your_data.tsv --features_path your_features.pt

  Or if you know target_std from training:
  python rescale_loss_plots.py --history_path unified_esm_results_corrected/training_history.json --output_dir unified_esm_results_corrected_rescaled --target_std 85.0
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_fig_multiformat(fig, basepath: str, dpi: int = 300) -> None:
    for ext in ("png", "pdf", "svg"):
        path = f"{basepath}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
    return


def main():
    parser = argparse.ArgumentParser(description="Rescale loss to original CCS scale and save plots")
    parser.add_argument("--history_path", type=str, required=True, help="Path to training_history.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save rescaled plots")
    parser.add_argument("--data_path", type=str, default=None, help="TSV (same as training) to compute target_std")
    parser.add_argument("--features_path", type=str, default=None, help=".pt file to align n with training")
    parser.add_argument("--target_std", type=float, default=None, help="Override: target_std from training (avoids loading data)")
    parser.add_argument("--charge_column", type=str, default="Charge")
    parser.add_argument("--target_column", type=str, default="CCS_Experimental")
    args = parser.parse_args()

    if not os.path.exists(args.history_path):
        raise FileNotFoundError(f"History not found: {args.history_path}")

    with open(args.history_path) as f:
        history = json.load(f)

    if args.target_std is not None:
        target_std = float(args.target_std)
        print(f"Using provided target_std = {target_std:.4f}")
    elif args.data_path and os.path.exists(args.data_path):
        import pandas as pd
        from unified_feature_embedding import stratified_split_indices_by_charge

        df = pd.read_csv(args.data_path, sep="\t")
        if args.charge_column not in df.columns or args.target_column not in df.columns:
            raise ValueError(f"TSV must have {args.charge_column} and {args.target_column}. Got {list(df.columns)}")
        charges_raw = np.asarray(df[args.charge_column].values, dtype=np.int64)
        targets_raw = np.asarray(df[args.target_column].values, dtype=np.float64)

        if args.features_path and os.path.exists(args.features_path):
            data = __import__("torch").load(args.features_path, weights_only=False)
            esm = data["esm_features"]
            n_feat = len(esm) if isinstance(esm, list) else esm.shape[0]
            n_tsv = len(df)
            n = min(n_feat, n_tsv)
            targets_raw = targets_raw[:n]
            charges_raw = charges_raw[:n]

        charges = np.clip(charges_raw.astype(np.int64) - 1, 0, None)
        train_idx, _, _ = stratified_split_indices_by_charge(
            charges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        train_targets = targets_raw[train_idx]
        target_std = float(np.std(train_targets))
        if target_std < 1e-8:
            target_std = 1.0
        print(f"Computed target_std from train split = {target_std:.4f}")
    else:
        raise ValueError("Provide either --target_std or both --data_path and (optionally) --features_path")

    train_loss_norm = np.array(history["train_loss"], dtype=np.float64)
    val_loss_norm = np.array(history["val_loss"], dtype=np.float64)
    train_r2 = history["train_r2"]
    val_r2 = history["val_r2"]

    train_loss_original = target_std * np.sqrt(np.maximum(train_loss_norm, 0.0))
    val_loss_original = target_std * np.sqrt(np.maximum(val_loss_norm, 0.0))

    os.makedirs(args.output_dir, exist_ok=True)

    rescaled_history = {
        "train_loss_normalized": history["train_loss"],
        "val_loss_normalized": history["val_loss"],
        "train_loss_original_scale_rmse": train_loss_original.tolist(),
        "val_loss_original_scale_rmse": val_loss_original.tolist(),
        "train_r2": history["train_r2"],
        "val_r2": history["val_r2"],
        "lr": history["lr"],
        "target_std_used": target_std,
    }
    with open(os.path.join(args.output_dir, "training_history_rescaled.json"), "w") as f:
        json.dump(rescaled_history, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(len(train_loss_original))
    ax[0].plot(epochs, train_loss_original, label="Train")
    ax[0].plot(epochs, val_loss_original, label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("RMSE (original CCS scale)")
    ax[0].legend()
    ax[0].set_title("Loss (RMSE, original scale)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, train_r2, label="Train")
    ax[1].plot(epochs, val_r2, label="Val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("R²")
    ax[1].legend()
    ax[1].set_title("R² (unchanged scale)")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig_multiformat(
        fig,
        os.path.join(args.output_dir, "unified_esm_training_curves_rescaled"),
        dpi=300,
    )
    plt.close(fig)

    print(f"Rescaled plots and training_history_rescaled.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()

