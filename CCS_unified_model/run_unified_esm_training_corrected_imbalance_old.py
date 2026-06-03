#!/usr/bin/env python3
"""
Unified ESM-2 + RNN CCS Training (corrected, with class imbalance handling)
============================================================================

Same as run_unified_esm_training_corrected.py except training uses charge-balanced
sampling so that all charges have fair, equal representation per epoch:

- WeightedRandomSampler: each training sample is weighted by 1 / (count of its charge).
  Minority charges are oversampled so that in expectation each charge is seen equally
  often. This prevents the model from underfitting rare charges (e.g. z=1).

- Stratified train/val/test split is unchanged (proportional per charge).
- No data leakage; CCS normalization from training set only.
- Val and test loaders remain unweighted (normal evaluation).

ReduceLROnPlateau on val loss is enabled by default (replaces per-batch cosine LR) to smooth
training and reduce overfitting; use --no_lr_scheduler for legacy cosine warmup only.
Optional --z1_boost > 1 increases sampling weight for charge z=1 (0-based index 0) to help
R² on the minority charge. Training curves in plots use a moving average when --plot_smooth_window > 0.
For heavy-tailed residuals / outliers on z=1, try --loss huber.

Usage:
  1. Extract features: run_extract_esm_for_unified.py or charge_aware_esm_feature_extraction
  2. Train:
     python run_unified_esm_training_corrected_imbalance.py --data_path your_data.tsv --features_path your_features.pt --output_dir ./unified_esm_results_corrected_imbalance

  To disable balanced sampling (ablation): add --no_balance_sampling
  Use the same --data_path and --features_path as run_unified_esm_training_corrected.py
  for a fair comparison. If output shows only some charges (e.g. 1–3), the input data
  contained only those charges; plots and metrics are per charge present in the test set.

HPC / "Too many open files": DataLoader workers use many file descriptors. If you see
RuntimeError from DataLoader, use --num_workers 0 or raise the shell limit, e.g. ulimit -n 8192
before launching Python. This script defaults persistent_workers=False to reduce FD use.

Performance: full train-set eval each epoch doubles work on huge datasets (~1e6+ samples).
Default --train_eval_max_batches=-1 matches len(val_loader) batches. When using charge-balanced
training, logged train loss/R² use a separate shuffle loader (natural train charge mix ≈ val),
not the weighted sampler—otherwise metrics are not comparable to val even if batch counts match.
Use 0 for full train (slow). R² uses one-pass streaming sums (no huge cat).
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reduce multiprocessing FD pressure on Linux clusters (safe no-op if unsupported).
try:
    import torch.multiprocessing as _torch_mp

    _torch_mp.set_sharing_strategy("file_system")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from unified_feature_embedding import stratified_split_indices_by_charge
from unified_esm_rnn_model import UnifiedESMCCSPredictor, ENGINEERED_FEAT_DIM


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    payload = {
        "sessionId": "f2f858",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("debug-f2f858.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


def get_lr_with_warmup_cosine(
    step: int, warmup_steps: int, total_steps: int, base_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def make_charge_balanced_sampler(
    train_indices: List[int],
    charges: np.ndarray,
    mode: str = "sqrt",
    z1_boost: float = 1.0,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Build a sampler that balances charge representation in training.

    mode="equal": weight = 1/count(charge) so each charge is equally likely per sample
      (strong oversampling of minority; can hurt overall R²).
    mode="sqrt": weight = 1/sqrt(count(charge)) for softer balance (minority seen more
      often but majority still dominant; often better overall R²).
    """
    train_indices = np.asarray(train_indices)
    train_charges = charges[train_indices]
    unique, counts = np.unique(train_charges, return_counts=True)
    charge_to_count = dict(zip(unique, counts))
    if mode == "sqrt":
        charge_to_weight = {c: 1.0 / (cnt ** 0.5) for c, cnt in charge_to_count.items()}
    else:
        charge_to_weight = {c: 1.0 / cnt for c, cnt in charge_to_count.items()}
    weights = np.array(
        [charge_to_weight[c] for c in train_charges],
        dtype=np.float64,
    )
    if z1_boost > 1.0:
        weights = apply_z1_boost_to_weights(weights, train_indices, charges, z1_boost)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )
    return sampler


def apply_z1_boost_to_weights(
    weights: np.ndarray,
    train_indices: np.ndarray,
    charges: np.ndarray,
    z1_boost: float,
) -> np.ndarray:
    """Multiply weights for z=1 (charge index 0) by z1_boost (>=1 upsamples z=1)."""
    if z1_boost <= 1.0:
        return weights
    out = weights.copy()
    tc = charges[train_indices]
    mask = tc == 0
    out[mask] *= z1_boost
    return out


def charge_sampling_distribution(
    train_indices: List[int],
    charges: np.ndarray,
    mode: str,
    z1_boost: float = 1.0,
) -> Dict[int, float]:
    """Theoretical P(sample has charge c) under WeightedRandomSampler weights (sums to 1)."""
    train_indices = np.asarray(train_indices)
    train_charges = charges[train_indices]
    unique, counts = np.unique(train_charges, return_counts=True)
    charge_to_count = dict(zip(unique, counts))
    if mode == "sqrt":
        charge_to_weight = {c: 1.0 / (cnt ** 0.5) for c, cnt in charge_to_count.items()}
    else:
        charge_to_weight = {c: 1.0 / cnt for c, cnt in charge_to_count.items()}
    w = np.array([charge_to_weight[c] for c in train_charges], dtype=np.float64)
    if z1_boost > 1.0:
        w = apply_z1_boost_to_weights(w, train_indices, charges, z1_boost)
    w_sum = w.sum()
    dist: Dict[int, float] = {}
    for c in unique:
        dist[int(c)] = float(w[train_charges == c].sum() / w_sum)
    return dist


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
        use_lr_scheduler: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
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
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )
        else:
            self.scheduler = None
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
        debug_charge_counts: Dict[int, int] = {}

        for batch_idx, (esm, eng, charge, target) in enumerate(train_loader):
            esm = esm.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            if epoch == 0 and batch_idx < 20:
                uq, ct = torch.unique(charge.detach(), return_counts=True)
                for u, c in zip(uq.tolist(), ct.tolist()):
                    debug_charge_counts[int(u)] = debug_charge_counts.get(int(u), 0) + int(c)

            if not self.use_lr_scheduler:
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

            if epoch == 0 and batch_idx == 19:
                # #region agent log
                _debug_log(
                    "overfit-run1",
                    "H3",
                    "run_unified_esm_training_corrected_imbalance.py:train_epoch",
                    "Observed early-batch charge mix from sampler",
                    {"first_batches": 20, "counts": debug_charge_counts},
                )
                # #endregion

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        max_batches: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Mean loss and global R² (eval mode). Uses streaming sums for R² (no full tensor concat).
        If max_batches is set, only the first N batches are used (for fast train metrics on huge data).
        """
        self.model.eval()
        non_blocking = self.device.type == "cuda"
        total_loss_num = 0.0
        total_loss_den = 0
        ss_res = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        n_total = 0
        tm, ts = self.target_mean, self.target_std

        for batch_idx, (esm, eng, charge, target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            esm = esm.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            pred = self.model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
            loss = criterion(pred, target)
            pred_raw = pred * ts + tm
            target_raw = target * ts + tm
            bs = int(target.numel())
            total_loss_num += float(loss.item()) * bs
            total_loss_den += bs
            diff = pred_raw - target_raw
            ss_res += float((diff * diff).sum().item())
            sum_y += float(target_raw.sum().item())
            sum_y2 += float((target_raw * target_raw).sum().item())
            n_total += bs

        mean_loss = total_loss_num / max(total_loss_den, 1)
        if n_total < 2:
            return mean_loss, 0.0
        mean_y = sum_y / n_total
        ss_tot = sum_y2 - n_total * (mean_y ** 2)
        r2_global = 1.0 - ss_res / max(ss_tot, 1e-8)
        return mean_loss, float(r2_global)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_path: Optional[str] = None,
        loss_type: str = "mse",
        val_every: int = 1,
        min_delta: float = 0.0,
        train_eval_max_batches: Optional[int] = None,
        train_metrics_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, List[float]]:
        criterion = nn.MSELoss() if loss_type == "mse" else nn.HuberLoss()
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(num_epochs):
            self.train_epoch(train_loader, criterion, epoch, num_epochs)

            # Train metrics: optional prefix only (full train eval is ~2x epoch time on huge data).
            # Use train_metrics_loader when set (e.g. natural shuffle) so curves match val charge mix.
            train_eval_limit = train_eval_max_batches if train_eval_max_batches else None
            metric_loader = (
                train_metrics_loader if train_metrics_loader is not None else train_loader
            )
            train_loss, train_r2 = self.evaluate(
                metric_loader, criterion, max_batches=train_eval_limit
            )
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

            if do_val and (epoch % 5 == 0 or epoch == num_epochs - 1):
                # #region agent log
                _debug_log(
                    "post-fix",
                    "H1",
                    "run_unified_esm_training_corrected_imbalance.py:train",
                    "Epoch-level generalization gap and scheduler state",
                    {
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "train_r2": float(train_r2),
                        "val_r2": float(val_r2),
                        "loss_gap": float(train_loss - val_loss),
                        "r2_gap": float(train_r2 - val_r2),
                        "lr": float(lr),
                        "use_lr_scheduler": bool(self.use_lr_scheduler),
                        "train_eval_max_batches": int(train_eval_max_batches) if train_eval_max_batches else 0,
                        "train_metrics_natural_mix": bool(train_metrics_loader is not None),
                    },
                )
                # #endregion

            if do_val:
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                if val_loss < best_val_loss - min_delta:
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


def _moving_average(series: List[float], window: int) -> List[float]:
    if window <= 1 or len(series) < window:
        return list(series)
    out: List[float] = []
    w = float(window)
    for i in range(len(series)):
        lo = max(0, i - window // 2)
        hi = min(len(series), lo + window)
        lo = max(0, hi - window)
        out.append(float(np.mean(series[lo:hi])))
    return out


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
    parser = argparse.ArgumentParser(
        description="Unified ESM-2 + RNN CCS training (corrected, charge-balanced sampling)"
    )
    parser.add_argument("--data_path", type=str, required=True, help="TSV with Charge, CCS_Experimental (same order as features)")
    parser.add_argument("--features_path", type=str, required=True, help=".pt from ESM + engineered feature extraction")
    parser.add_argument("--output_dir", type=str, default="./unified_esm_results_corrected_imbalance")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--min_delta",
        type=float,
        default=5e-5,
        help="Min val_loss improvement for early stopping / best checkpoint",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial LR (lower default with ReduceLROnPlateau)")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--dropout", type=float, default=0.4, help="Model dropout (higher reduces overfitting)")
    parser.add_argument("--weight_decay", type=float, default=2e-4, help="AdamW weight decay")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers; use 0 on HPC if you hit 'Too many open files'",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (faster, uses more file descriptors)",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--esm_dim", type=int, default=None, help="Infer from .pt if not set")
    parser.add_argument(
        "--no_balance_sampling",
        action="store_true",
        help="Disable charge-balanced sampling (train with plain shuffle for ablation)",
    )
    parser.add_argument(
        "--balance_mode",
        type=str,
        default="sqrt",
        choices=["equal", "sqrt"],
        help="equal: each charge equally likely per sample (strong; can hurt overall R²). sqrt: softer (default, often better R²).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="RNG seed for train/val/test split (use same as corrected run for fair comparison)",
    )
    parser.add_argument(
        "--z1_boost",
        type=float,
        default=1.25,
        help="Sampler weight multiplier for z=1 only (>1 upsamples z=1 further; try 1.5 if R² z=1 still low). 1.0 disables.",
    )
    parser.add_argument(
        "--no_lr_scheduler",
        action="store_true",
        help="Use legacy cosine LR per batch instead of ReduceLROnPlateau on val loss",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Epochs without val_loss improvement before LR reduction",
    )
    parser.add_argument(
        "--plot_smooth_window",
        type=int,
        default=5,
        help="Moving-average window for training curve plots (1 = no smoothing)",
    )
    parser.add_argument(
        "--train_eval_max_batches",
        type=int,
        default=-1,
        help="-1 (default): use len(val_loader) batches for train metrics (same ~sample count as val). 0 = full train (slow). >0 = fixed batch cap.",
    )
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

    # Diagnostic: charges present in full data (so user can confirm same data as corrected run)
    uniq_all, cnt_all = np.unique(charges, return_counts=True)
    charge_summary = " ".join([f"z={int(u)+1}:{c}" for u, c in zip(uniq_all, cnt_all)])
    print(f"Loaded {n_samples} samples. Charges in data: {charge_summary}")

    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        charges,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.split_seed,
    )
    # Per-charge split counts (1-based charge labels)
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ch_arr = charges[np.asarray(idx)]
        uniq, cnt = np.unique(ch_arr, return_counts=True)
        msg = " ".join([f"z={int(u)+1}:{c}" for u, c in zip(uniq, cnt)])
        print(f"  {split_name}: {msg}")
    # #region agent log
    _debug_log(
        "overfit-run1",
        "H2",
        "run_unified_esm_training_corrected_imbalance.py:main",
        "Dataset and split charge distribution before training",
        {
            "n_samples": int(n_samples),
            "charge_counts_all": {int(u): int(c) for u, c in zip(uniq_all.tolist(), cnt_all.tolist())},
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "split_seed": int(args.split_seed),
        },
    )
    # #endregion
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
    # persistent_workers=True keeps worker processes open and can exhaust ulimit -n on HPC.
    loader_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda and args.num_workers > 0,
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 4
    if args.num_workers > 0 and not args.persistent_workers:
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(
                f"DataLoader: num_workers={args.num_workers}, persistent_workers=False "
                f"(open file soft limit={soft}; raise with ulimit -n if needed)"
            )
        except Exception:
            print(
                f"DataLoader: num_workers={args.num_workers}, persistent_workers=False"
            )
    balance_sampling = not args.no_balance_sampling
    if balance_sampling:
        dist = charge_sampling_distribution(
            train_idx, charges, args.balance_mode, z1_boost=args.z1_boost
        )
        dist_msg = " ".join([f"P(z={k+1})={v:.3f}" for k, v in sorted(dist.items())])
        print(f"Sampler target charge mix (theoretical): {dist_msg}")
        # #region agent log
        _debug_log(
            "overfit-run1",
            "H3",
            "run_unified_esm_training_corrected_imbalance.py:main",
            "Sampler configuration and theoretical charge probabilities",
            {
                "balance_mode": args.balance_mode,
                "z1_boost": float(args.z1_boost),
                "theoretical_probabilities": {int(k): float(v) for k, v in dist.items()},
            },
        )
        # #endregion
        sampler = make_charge_balanced_sampler(
            train_idx,
            charges,
            mode=args.balance_mode,
            z1_boost=args.z1_boost,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, sampler=sampler, shuffle=False, **loader_kw
        )
        print(
            f"Training with charge-balanced sampling (mode={args.balance_mode}, z1_boost={args.z1_boost})."
        )
    else:
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kw)
        print("Training with plain shuffle (no balance).")
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_kw)

    # Logged train metrics must not use the weighted sampler, or loss/R² vs val reflect different charge mixes.
    train_metrics_loader: Optional[torch.utils.data.DataLoader] = None
    if balance_sampling:
        train_metrics_loader = torch.utils.data.DataLoader(
            train_set, shuffle=True, **loader_kw
        )
        ch_val = charges[np.asarray(val_idx)]
        uniq_v, cnt_v = np.unique(ch_val, return_counts=True)
        val_frac = {
            int(u): float(c) / max(len(val_idx), 1) for u, c in zip(uniq_v, cnt_v)
        }
        sampler_tgt = {int(k): float(v) for k, v in sorted(dist.items())}
        # #region agent log
        _debug_log(
            "metric-mix",
            "H5",
            "run_unified_esm_training_corrected_imbalance.py:main",
            "Val empirical charge fraction vs balanced sampler target (root of train/val metric mismatch)",
            {
                "val_fraction_by_charge_idx": val_frac,
                "sampler_target_fraction": sampler_tgt,
                "train_metrics_loader": "natural_shuffle_on_train_subset",
            },
        )
        # #endregion

    if args.train_eval_max_batches < 0:
        train_eval_effective_batches = len(val_loader)
    elif args.train_eval_max_batches == 0:
        train_eval_effective_batches = None
    else:
        train_eval_effective_batches = int(args.train_eval_max_batches)
    if train_eval_effective_batches is not None:
        approx_n = train_eval_effective_batches * args.batch_size
        mix_note = (
            " (natural charge mix via shuffle; same as val proportions in expectation)"
            if train_metrics_loader is not None
            else ""
        )
        print(
            f"Train metrics each epoch: first {train_eval_effective_batches} batches from "
            f"{'a separate shuffled train loader' if train_metrics_loader is not None else 'train loader'}"
            f"{mix_note}, ~{approx_n} samples; val loader has {len(val_loader)} batches."
        )
    else:
        print("Train metrics each epoch: FULL train set (slow on large datasets).")
    # #region agent log
    _debug_log(
        "post-fix",
        "H1",
        "run_unified_esm_training_corrected_imbalance.py:main",
        "Resolved train-eval batch cap for fair train vs val metrics",
        {
            "train_eval_arg": int(args.train_eval_max_batches),
            "effective_train_eval_batches": train_eval_effective_batches,
            "val_loader_len_batches": int(len(val_loader)),
            "val_samples_approx": int(len(val_idx)),
            "train_metrics_natural_shuffle": train_metrics_loader is not None,
        },
    )
    # #endregion

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
        dropout=args.dropout,
        output_dim=1,
    )

    use_sched = not args.no_lr_scheduler
    trainer = UnifiedESMTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=args.amp,
        use_lr_scheduler=use_sched,
        scheduler_patience=args.scheduler_patience,
    )
    if use_sched:
        print("Using ReduceLROnPlateau on val loss (no per-batch cosine LR).")
    else:
        print("Using cosine warmup LR per batch (--no_lr_scheduler).")

    save_path = os.path.join(args.output_dir, "best_unified_esm_model.pt")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=save_path,
        loss_type=args.loss,
        val_every=args.val_every,
        min_delta=args.min_delta,
        train_eval_max_batches=train_eval_effective_batches,
        train_metrics_loader=train_metrics_loader,
    )

    history_out = {
        **history,
        "train_eval_max_batches_arg": args.train_eval_max_batches,
        "train_eval_max_batches_effective": train_eval_effective_batches,
        "train_metrics_natural_shuffle": train_metrics_loader is not None,
        "balanced_sampling": balance_sampling,
        "balance_mode": args.balance_mode if balance_sampling else None,
        "split_seed": args.split_seed,
        "z1_boost": args.z1_boost if balance_sampling else None,
        "use_lr_scheduler": use_sched,
        "scheduler_patience": args.scheduler_patience if use_sched else None,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "plot_smooth_window": args.plot_smooth_window,
    }
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history_out, f, indent=2)

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
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    per_ch = charge_stratified_metrics(preds, targets_test, charges_test)
    per_ch_display = {str(int(k) + 1): v for k, v in per_ch.items()}
    z1_mask = charges_test == 0
    z1_slope = None
    if np.sum(z1_mask) > 1:
        z1_slope = float(np.polyfit(targets_test[z1_mask], (preds - targets_test)[z1_mask], 1)[0])
    # #region agent log
    _debug_log(
        "overfit-run1",
        "H4",
        "run_unified_esm_training_corrected_imbalance.py:main",
        "Final test metrics and z1 residual trend",
        {
            "overall_r2": float(r2_score(targets_test, preds)),
            "overall_rmse": float(np.sqrt(mean_squared_error(targets_test, preds))),
            "z1_r2": float(per_ch.get(0, {}).get("r2", float("nan"))),
            "z1_rmse": float(per_ch.get(0, {}).get("rmse", float("nan"))),
            "z1_residual_vs_target_slope": z1_slope,
        },
    )
    # #endregion
    with open(os.path.join(args.output_dir, "charge_stratified_metrics.json"), "w") as f:
        json.dump(per_ch_display, f, indent=2)

    print("\nUnified ESM model (corrected, imbalance-aware) – test set:")
    print(f"  RMSE = {np.sqrt(mean_squared_error(targets_test, preds)):.4f}")
    print(f"  MAE  = {mean_absolute_error(targets_test, preds):.4f}")
    print(f"  R²   = {r2_score(targets_test, preds):.4f}")
    print("Per charge:", per_ch_display)

    plot_charge_stratified(preds, targets_test, charges_test, args.output_dir)

    w = max(1, int(args.plot_smooth_window))
    tr_l = _moving_average(history["train_loss"], w)
    va_l = _moving_average(history["val_loss"], w)
    tr_r = _moving_average(history["train_r2"], w)
    va_r = _moving_average(history["val_r2"], w)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(tr_l, label="Train (smoothed)", color="C0", linewidth=1.8)
    ax[0].plot(va_l, label="Val (smoothed)", color="C1", linewidth=1.8)
    if w > 1:
        ax[0].plot(history["train_loss"], label="Train (raw)", color="C0", alpha=0.25, linewidth=0.8)
        ax[0].plot(history["val_loss"], label="Val (raw)", color="C1", alpha=0.25, linewidth=0.8)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend(fontsize=8)
    title_loss = "Loss (eval mode)"
    if w > 1:
        title_loss += f", MA window={w}"
    ax[0].set_title(title_loss)
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(tr_r, label="Train (smoothed)", color="C0", linewidth=1.8)
    ax[1].plot(va_r, label="Val (smoothed)", color="C1", linewidth=1.8)
    if w > 1:
        ax[1].plot(history["train_r2"], label="Train (raw)", color="C0", alpha=0.25, linewidth=0.8)
        ax[1].plot(history["val_r2"], label="Val (raw)", color="C1", alpha=0.25, linewidth=0.8)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("R² (global, full set)")
    ax[1].legend(fontsize=8)
    title_r2 = "R² (eval mode, global)"
    if w > 1:
        title_r2 += f", MA window={w}"
    ax[1].set_title(title_r2)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig_multiformat(fig, os.path.join(args.output_dir, "unified_esm_training_curves"), dpi=300)
    plt.close(fig)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

