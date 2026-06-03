#!/usr/bin/env python3
"""
Unified ESM-2 + RNN CCS Training and Evaluation (corrected)
============================================================

Corrected version of run_unified_esm_training.py:

- Train and validation metrics are both computed in eval mode (model.eval(), no dropout)
  so that training-set and validation-set curves are directly comparable. This fixes
  the misleading pattern where validation loss appeared lower than training loss
  (previously training loss was computed with dropout active).
- No data leakage: CCS normalization (target_mean, target_std) is computed from the
  training set only. Train/val/test splits are stratified by charge and do not overlap.
- Same features as the original: ESM-2 + engineered features, stratified split,
  early stopping, optional AMP, multi-format plots.

Usage:
  1. Extract features: run_extract_esm_for_unified.py or charge_aware_esm_feature_extraction
  2. Train:
     python run_unified_esm_training_corrected.py --data_path your_data.tsv --features_path your_features.pt --output_dir ./unified_esm_results_corrected
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from unified_feature_embedding import stratified_split_indices_by_charge
from unified_esm_rnn_model import UnifiedESMCCSPredictor, ENGINEERED_FEAT_DIM


def get_lr_with_warmup_cosine(
    step: int, warmup_steps: int, total_steps: int, base_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


class UnifiedESMTrainer:
    def __init__(
        self,
        model: UnifiedESMCCSPredictor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_ratio: float = 0.1,
        gradient_clip: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        device: str = "auto",
        use_amp: bool = False,
    ):
        self.model = model
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.base_lr = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_clip = gradient_clip
        self.target_mean = target_mean
        self.target_std = target_std
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_r2": [], "val_r2": [], "lr": [],
        }

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
    ) -> None:
        """Run one training epoch (gradient updates). Does not return metrics."""
        self.model.train()
        total_steps = total_epochs * len(train_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)
        non_blocking = self.device.type == "cuda"

        for batch_idx, (esm, eng, charge, target) in enumerate(train_loader):
            esm = esm.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)

            step = epoch * len(train_loader) + batch_idx
            lr = get_lr_with_warmup_cosine(step, warmup_steps, total_steps, self.base_lr)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
                    loss = criterion(pred, target)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(
        self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Compute loss and R² in eval mode (no dropout). Use for both train and val reporting."""
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        non_blocking = self.device.type == "cuda"

        for esm, eng, charge, target in data_loader:
            esm = esm.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            pred = self.model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
            loss = criterion(pred, target)
            pred_raw = pred * self.target_std + self.target_mean
            target_raw = target * self.target_std + self.target_mean
            r2 = _r2_torch(pred_raw, target_raw)
            total_loss += loss.item()
            total_r2 += r2
            n_batches += 1

        return total_loss / max(n_batches, 1), total_r2 / max(n_batches, 1)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_path: Optional[str] = None,
        loss_type: str = "mse",
        val_every: int = 1,
    ) -> Dict[str, List[float]]:
        criterion = nn.MSELoss() if loss_type == "mse" else nn.HuberLoss()
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(num_epochs):
            self.train_epoch(train_loader, criterion, epoch, num_epochs)

            # Report train and val metrics in eval mode for fair comparison (no dropout)
            train_loss, train_r2 = self.evaluate(train_loader, criterion)
            do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
            if do_val:
                val_loss, val_r2 = self.evaluate(val_loader, criterion)
            else:
                val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else float("inf")
                val_r2 = self.history["val_r2"][-1] if self.history["val_r2"] else 0.0

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_r2"].append(train_r2)
            self.history["val_r2"].append(val_r2)
            self.history["lr"].append(lr)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  lr={lr:.2e}")

            if do_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        return self.history


def _r2_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def charge_stratified_metrics(
    preds: np.ndarray, targets: np.ndarray, charges: np.ndarray
) -> Dict[int, Dict[str, float]]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    out = {}
    for ch in np.unique(charges):
        mask = charges == ch
        if mask.sum() < 2:
            continue
        p, t = preds[mask], targets[mask]
        out[int(ch)] = {
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "mae": float(mean_absolute_error(t, p)),
            "r2": float(r2_score(t, p)),
            "n": int(mask.sum()),
        }
    return out


def _save_fig_multiformat(fig: "plt.Figure", basepath: str, dpi: int = 300) -> None:
    """Save figure as high-resolution PNG, PDF, and SVG (basepath without extension)."""
    for ext in ("png", "pdf", "svg"):
        path = f"{basepath}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
    return


def plot_charge_stratified(
    preds: np.ndarray,
    targets: np.ndarray,
    charges: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import r2_score

    ch_list = sorted(np.unique(charges))
    ch_labels = [str(int(c) + 1) for c in ch_list]
    residuals = preds - targets

    # Combined plots (all charges)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].scatter(targets, preds, alpha=0.6, s=20)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2)
    axes[0].set_xlabel("Experimental CCS")
    axes[0].set_ylabel("Predicted CCS")
    axes[0].set_title(f"Predicted vs Experimental (R² = {r2_score(targets, preds):.4f})")
    axes[0].grid(True, alpha=0.3)

    for ch in ch_list:
        mask = charges == ch
        axes[1].scatter(preds[mask], residuals[mask], alpha=0.5, label=f"z={int(ch)+1}", s=15)
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Predicted CCS")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual by charge")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Residual distribution")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig_multiformat(fig, os.path.join(output_dir, "unified_esm_evaluation_plots"), dpi=300)
    plt.close(fig)

    # Per-charge plots
    for ch in ch_list:
        mask = charges == ch
        t_ch = targets[mask]
        p_ch = preds[mask]
        r_ch = p_ch - t_ch
        z_label = int(ch) + 1
        n_ch = mask.sum()

        fig_ch, ax_ch = plt.subplots(1, 3, figsize=(14, 5))
        fig_ch.suptitle(f"Unified ESM – Charge z={z_label} (n={n_ch})", fontsize=12, fontweight="bold")

        ax_ch[0].scatter(t_ch, p_ch, alpha=0.6, s=20)
        ax_ch[0].plot([t_ch.min(), t_ch.max()], [t_ch.min(), t_ch.max()], "r--", lw=2)
        ax_ch[0].set_xlabel("Experimental CCS")
        ax_ch[0].set_ylabel("Predicted CCS")
        ax_ch[0].set_title(f"Experimental vs Predicted (R² = {r2_score(t_ch, p_ch):.4f})")
        ax_ch[0].grid(True, alpha=0.3)

        ax_ch[1].scatter(t_ch, r_ch, alpha=0.6, s=20)
        ax_ch[1].axhline(0, color="k", linestyle="--", alpha=0.7)
        ax_ch[1].set_xlabel("Experimental CCS")
        ax_ch[1].set_ylabel("Residual")
        ax_ch[1].set_title("Experimental vs Residual")
        ax_ch[1].grid(True, alpha=0.3)

        ax_ch[2].hist(r_ch, bins=min(50, max(10, len(r_ch) // 5)), edgecolor="black", alpha=0.7)
        ax_ch[2].set_xlabel("Residual")
        ax_ch[2].set_ylabel("Frequency")
        ax_ch[2].set_title("Residual distribution")
        ax_ch[2].grid(True, alpha=0.3)

        plt.tight_layout()
        _save_fig_multiformat(
            fig_ch,
            os.path.join(output_dir, f"unified_esm_charge_z{z_label}_plots"),
            dpi=300,
        )
        plt.close(fig_ch)


# ---------------------------------------------------------------------------
# Dataset and data loading
# ---------------------------------------------------------------------------


class UnifiedESMDataset(torch.utils.data.Dataset):
    """Dataset of (pooled_esm, engineered, charge_0based, ccs_normalized)."""

    def __init__(
        self,
        esm_tensor: torch.Tensor,
        engineered_tensor: torch.Tensor,
        charges: np.ndarray,
        targets: np.ndarray,
        target_mean: float,
        target_std: float,
    ):
        assert len(esm_tensor) == len(engineered_tensor) == len(charges) == len(targets)
        self.esm = esm_tensor
        self.engineered = engineered_tensor
        self.charges = torch.tensor(charges, dtype=torch.long)
        self.targets_raw = torch.tensor(targets, dtype=torch.float32)
        self.target_mean = target_mean
        self.target_std = target_std
        self.targets = (self.targets_raw - target_mean) / target_std

    def __len__(self) -> int:
        return len(self.esm)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.esm[idx],
            self.engineered[idx],
            self.charges[idx],
            self.targets[idx],
        )


def load_features_and_tsv(
    features_path: str,
    data_path: str,
    sequence_column: str = "Sequence",
    charge_column: str = "Charge",
    target_column: str = "CCS_Experimental",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load .pt (esm_features, engineered_features) and TSV (Charge, CCS).
    Align by row index; trim to min length. Charges returned 0-based for embedding.
    """
    data = torch.load(features_path, weights_only=False)
    esm_features = data["esm_features"]
    engineered_features = data["engineered_features"]

    if isinstance(esm_features, list):
        esm_tensor = torch.stack([f.cpu() if f.is_cuda else f for f in esm_features])
    else:
        esm_tensor = esm_features.cpu() if esm_features.is_cuda else esm_features

    if isinstance(engineered_features, list):
        eng_tensor = torch.tensor(np.array(engineered_features), dtype=torch.float32)
    else:
        eng_tensor = torch.tensor(np.asarray(engineered_features), dtype=torch.float32)

    df = pd.read_csv(data_path, sep="\t")
    if charge_column not in df.columns or target_column not in df.columns:
        raise ValueError(f"TSV must contain {charge_column} and {target_column}. Got {list(df.columns)}")

    charges_raw = np.asarray(df[charge_column].values, dtype=np.int64)
    targets_raw = np.asarray(df[target_column].values, dtype=np.float64)

    n_feat = len(esm_tensor)
    n_tsv = len(df)
    if n_feat != n_tsv:
        print(f"Aligning by index: features n={n_feat}, TSV n={n_tsv}; using min={min(n_feat, n_tsv)}")
    n = min(n_feat, n_tsv)
    esm_tensor = esm_tensor[:n]
    eng_tensor = eng_tensor[:n]
    charges = np.clip(charges_raw[:n].astype(np.int64) - 1, 0, None)
    targets = targets_raw[:n]
    return esm_tensor, eng_tensor, charges, targets


def main():
    parser = argparse.ArgumentParser(description="Unified ESM-2 + RNN CCS training (corrected)")
    parser.add_argument("--data_path", type=str, required=True, help="TSV with Charge, CCS_Experimental (same order as features)")
    parser.add_argument("--features_path", type=str, required=True, help=".pt from ESM + engineered feature extraction")
    parser.add_argument("--output_dir", type=str, default="./unified_esm_results_corrected")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--esm_dim", type=int, default=None, help="Infer from .pt if not set")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return
    if not os.path.exists(args.features_path):
        print(f"Features not found: {args.features_path}. Run ESM feature extraction first.")
        return

    esm_tensor, eng_tensor, charges, targets = load_features_and_tsv(
        args.features_path, args.data_path
    )
    n_samples = len(esm_tensor)
    esm_dim = esm_tensor.shape[1]
    eng_dim = eng_tensor.shape[1]
    if args.esm_dim is not None and args.esm_dim != esm_dim:
        print(f"Warning: --esm_dim {args.esm_dim} != inferred {esm_dim}; using {esm_dim}")

    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        charges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    train_targets = targets[train_idx]
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets))
    if target_std < 1e-8:
        target_std = 1.0

    full_dataset = UnifiedESMDataset(
        esm_tensor, eng_tensor, charges, targets,
        target_mean=target_mean, target_std=target_std,
    )
    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set = torch.utils.data.Subset(full_dataset, val_idx)
    test_set = torch.utils.data.Subset(full_dataset, test_idx)

    use_cuda = torch.cuda.is_available()
    loader_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda and args.num_workers > 0,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_kw)

    num_charges = max(10, int(charges.max()) + 1)
    model = UnifiedESMCCSPredictor(
        esm_dim=esm_dim,
        engineered_feat_dim=eng_dim,
        num_charges=num_charges,
        charge_embed_dim=32,
        rnn_input_dim=256,
        rnn_hidden_dim=128,
        num_layers=2,
        rnn_type=args.rnn_type,
        dropout=0.3,
        output_dim=1,
    )

    trainer = UnifiedESMTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=1e-5,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=args.amp,
    )

    save_path = os.path.join(args.output_dir, "best_unified_esm_model.pt")
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=save_path,
        loss_type=args.loss,
        val_every=args.val_every,
    )

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
    device = trainer.device
    model.to(device)

    # Evaluate on test set
    model.eval()
    preds_list, targets_list, charges_list = [], [], []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for esm, eng, charge, target in test_loader:
            esm = esm.to(device, non_blocking=non_blocking)
            eng = eng.to(device, non_blocking=non_blocking)
            charge = charge.to(device, non_blocking=non_blocking)
            out = model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
            pred_raw = out.cpu().numpy() * target_std + target_mean
            target_raw = target.cpu().numpy() * target_std + target_mean
            preds_list.append(pred_raw)
            targets_list.append(target_raw)
            charges_list.append(charge.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets_test = np.concatenate(targets_list)
    charges_test = np.concatenate(charges_list)

    per_ch = charge_stratified_metrics(preds, targets_test, charges_test)
    per_ch_display = {str(int(k) + 1): v for k, v in per_ch.items()}
    with open(os.path.join(args.output_dir, "charge_stratified_metrics.json"), "w") as f:
        json.dump(per_ch_display, f, indent=2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print("\nUnified ESM model (corrected) – test set:")
    print(f"  RMSE = {np.sqrt(mean_squared_error(targets_test, preds)):.4f}")
    print(f"  MAE  = {mean_absolute_error(targets_test, preds):.4f}")
    print(f"  R²   = {r2_score(targets_test, preds):.4f}")
    print("Per charge:", per_ch_display)

    plot_charge_stratified(preds, targets_test, charges_test, args.output_dir)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="Train")
    ax[0].plot(history["val_loss"], label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_title("Loss (eval mode, comparable)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["train_r2"], label="Train")
    ax[1].plot(history["val_r2"], label="Val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("R²")
    ax[1].legend()
    ax[1].set_title("R² (eval mode, comparable)")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig_multiformat(fig, os.path.join(args.output_dir, "unified_esm_training_curves"), dpi=300)
    plt.close(fig)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

