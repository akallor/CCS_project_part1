#!/usr/bin/env python3
"""
Unified RNN Training and Evaluation
===================================

- Loads data via unified_feature_embedding (sequence, charge, optional aux).
- Stratified train/val/test split by charge.
- Trains UnifiedRNNCCSPredictor with mixed-charge batches.
- Loss: MSE or Huber; optimizer: AdamW; scheduler: warmup + cosine decay.
- Early stopping on validation loss.
- Evaluation stratified by charge: boxplot, violin, scatter (length vs performance),
  residual plot by charge; retains previous-format plots (pred vs exp, residual dist).
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from unified_feature_embedding import (
    get_vocab_size,
    get_pad_idx,
    load_tsv_for_unified,
    UnifiedPeptideDataset,
    stratified_split_indices_by_charge,
    stratified_split_by_charge,
    collate_unified_batch,
)
from unified_rnn_model import UnifiedRNNCCSPredictor


def get_lr_with_warmup_cosine(
    step: int, warmup_steps: int, total_steps: int, base_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


class UnifiedTrainer:
    def __init__(
        self,
        model: UnifiedRNNCCSPredictor,
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
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "lr": [],
        }

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        total_steps = total_epochs * len(train_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)

        non_blocking = self.device.type == "cuda"
        for batch_idx, (seq, charge, aux, target, lengths) in enumerate(train_loader):
            seq = seq.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            aux = aux.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)

            step = epoch * len(train_loader) + batch_idx
            lr = get_lr_with_warmup_cosine(
                step, warmup_steps, total_steps, self.base_lr
            )
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(sequence=seq, charge=charge, aux=aux, lengths=lengths).squeeze(-1)
                    loss = criterion(pred, target)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(sequence=seq, charge=charge, aux=aux, lengths=lengths).squeeze(-1)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            pred_raw = pred.detach() * self.target_std + self.target_mean
            target_raw = target * self.target_std + self.target_mean
            r2 = _r2_torch(pred_raw, target_raw)
            total_loss += loss.item()
            total_r2 += r2
            n_batches += 1

        return total_loss / max(n_batches, 1), total_r2 / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        non_blocking = self.device.type == "cuda"
        for seq, charge, aux, target, lengths in val_loader:
            seq = seq.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            aux = aux.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            pred = self.model(sequence=seq, charge=charge, aux=aux, lengths=lengths).squeeze(-1)
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
            train_loss, train_r2 = self.train_epoch(train_loader, criterion, epoch, num_epochs)
            do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
            if do_val:
                val_loss, val_r2 = self.validate(val_loader, criterion)
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
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    return r2


def cosine_similarity_per_psm(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample cosine similarity (for vector outputs); for scalar CCS, return correlation-like score per sample (same as normalized residual)."""
    if pred.ndim == 1 and target.ndim == 1:
        # Scalar: use 1 - normalized absolute error as proxy for "agreement"
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
    dot = (pred * target).sum(axis=1)
    norm_p = np.linalg.norm(pred, axis=1) + 1e-8
    norm_t = np.linalg.norm(target, axis=1) + 1e-8
    return (dot / (norm_p * norm_t)).astype(np.float64)


def evaluate_model(
    model: UnifiedRNNCCSPredictor,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns predictions, targets, charges, seq_lengths, cosine_sim (per sample)."""
    model.eval()
    preds, targets, charges, seq_lengths = [], [], [], []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for seq, charge, aux, target, lengths in data_loader:
            seq = seq.to(device, non_blocking=non_blocking)
            charge = charge.to(device, non_blocking=non_blocking)
            aux = aux.to(device, non_blocking=non_blocking)
            out = model(sequence=seq, charge=charge, aux=aux, lengths=lengths).squeeze(-1)
            pred_raw = out.cpu().numpy() * target_std + target_mean
            target_raw = target.cpu().numpy() * target_std + target_mean
            preds.append(pred_raw)
            targets.append(target_raw)
            charges.append(charge.cpu().numpy())
            seq_lengths.append(lengths.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    charges = np.concatenate(charges)
    seq_lengths = np.concatenate(seq_lengths)
    # Per-PSM "cosine" for scalar: use correlation over batch or per-sample agreement
    cos_sim = cosine_similarity_per_psm(preds.reshape(-1, 1), targets.reshape(-1, 1)).flatten()
    return preds, targets, charges, seq_lengths, cos_sim


def charge_stratified_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    charges: np.ndarray,
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


def plot_charge_stratified(
    preds: np.ndarray,
    targets: np.ndarray,
    charges: np.ndarray,
    seq_lengths: np.ndarray,
    cos_sim: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import r2_score

    # Charges in batch are 0-based (embedding index); display as 1-based (z=1,2,3)
    ch_list = sorted(np.unique(charges))
    ch_labels = [str(int(c) + 1) for c in ch_list]

    # 1) Boxplot: charge vs cosine similarity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Unified Model – Charge-Stratified Evaluation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    data = [cos_sim[charges == c] for c in ch_list]
    ax.boxplot(data, labels=ch_labels)
    ax.set_xlabel("Charge state")
    ax.set_ylabel("Cosine similarity (agreement)")
    ax.set_title("Boxplot per charge")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.violinplot(
        data, positions=range(len(ch_list)), showmeans=True, showmedians=True
    )
    ax.set_xticks(range(len(ch_list)))
    ax.set_xticklabels(ch_labels)
    ax.set_xlabel("Charge state")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Violin per charge")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for ch in ch_list:
        mask = charges == ch
        ax.scatter(seq_lengths[mask], cos_sim[mask], alpha=0.5, label=f"z={int(ch)+1}", s=15)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Length vs performance by charge")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    residuals = preds - targets
    for ch in ch_list:
        mask = charges == ch
        ax.scatter(preds[mask], residuals[mask], alpha=0.5, label=f"z={int(ch)+1}", s=15)
    ax.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Predicted CCS")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs predicted (by charge)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unified_charge_stratified_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Residual plot stratified by charge (facets)
    ncharges = len(ch_list)
    ncol = min(3, ncharges)
    nrow = (ncharges + ncol - 1) // ncol
    fig3, axes3 = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    if ncharges == 1:
        axes3 = np.array([axes3])
    axes3 = axes3.flatten()
    residuals = preds - targets
    for i, ch in enumerate(ch_list):
        mask = charges == ch
        ax = axes3[i]
        ax.scatter(preds[mask], residuals[mask], alpha=0.6, s=20)
        ax.axhline(0, color="k", linestyle="--", alpha=0.7)
        ax.set_xlabel("Predicted CCS")
        ax.set_ylabel("Residual")
        ax.set_title(f"Charge z={int(ch)+1} (n={mask.sum()})")
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unified_residual_by_charge_facets.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Retain previous-format plots: Predicted vs Experimental, Residual, Residual dist
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    axes2[0].scatter(targets, preds, alpha=0.6, s=20)
    axes2[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2)
    axes2[0].set_xlabel("Experimental CCS")
    axes2[0].set_ylabel("Predicted CCS")
    axes2[0].set_title(f"Predicted vs Experimental (R² = {r2_score(targets, preds):.4f})")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].scatter(targets, preds - targets, alpha=0.6, s=20)
    axes2[1].axhline(0, color="r", linestyle="--", lw=2)
    axes2[1].set_xlabel("Experimental CCS")
    axes2[1].set_ylabel("Residual")
    axes2[1].set_title("Residual plot")
    axes2[1].grid(True, alpha=0.3)

    axes2[2].hist(preds - targets, bins=50, edgecolor="black", alpha=0.7)
    axes2[2].set_xlabel("Residual")
    axes2[2].set_ylabel("Frequency")
    axes2[2].set_title("Residual distribution")
    axes2[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unified_evaluation_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Unified RNN CCS training and evaluation")
    parser.add_argument("--data_path", type=str, default=None, help="TSV with Sequence, Charge, CCS_Experimental")
    parser.add_argument("--test_path", type=str, default=None, help="Optional test TSV")
    parser.add_argument("--output_dir", type=str, default="./unified_results")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--aux_columns", type=str, nargs="*", default=None, help="e.g. CCS_Experimental inv_K0 RT")
    parser.add_argument("--batch_size", type=int, default=128, help="Larger batches improve GPU utilization (e.g. 128–256)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for faster data loading (0 = main process only)")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (FP16) for faster GPU training")
    parser.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs (1 = every epoch)")
    args = parser.parse_args()

    # Default paths if not provided (use a single TSV with all charges for unified training)
    data_path = args.data_path or os.path.join(os.path.dirname(__file__), "train_1_new_charge1_lab.tsv")
    test_path = args.test_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}. Please provide --data_path.")
        print("For unified training, use a TSV with columns Sequence, Charge, CCS_Experimental containing all charge states (1,2,3).")
        return

    # Infer column names from first file
    df_sample = pd.read_csv(data_path, sep="\t", nrows=1)
    seq_col = "Sequence" if "Sequence" in df_sample.columns else df_sample.columns[1]
    ch_col = "Charge" if "Charge" in df_sample.columns else "Charge"
    tgt_col = "CCS_Experimental" if "CCS_Experimental" in df_sample.columns else "CCS_Experimental"

    sequences, charges, targets, aux_features = load_tsv_for_unified(
        data_path,
        sequence_column=seq_col,
        charge_column=ch_col,
        target_column=tgt_col,
        aux_columns=args.aux_columns,
        max_length=args.max_length,
    )
    n_samples = len(sequences)
    targets = np.asarray(targets, dtype=np.float64)
    charges = np.asarray(charges)
    aux_dim = aux_features.shape[1] if aux_features is not None else 0

    # Stratified indices (from charge array) so we can compute train-only mean/std
    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        charges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    train_targets = targets[train_idx]
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets))
    if target_std < 1e-8:
        target_std = 1.0

    full_dataset = UnifiedPeptideDataset(
        sequences=sequences,
        charges=charges,
        targets=targets,
        max_length=args.max_length,
        aux_features=aux_features,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=True,
    )
    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set = torch.utils.data.Subset(full_dataset, val_idx)

    if test_path and os.path.exists(test_path):
        # Load separate test set (same normalization as train)
        seq_test, ch_test, tgt_test, aux_test = load_tsv_for_unified(
            test_path, sequence_column=seq_col, charge_column=ch_col,
            target_column=tgt_col, aux_columns=args.aux_columns, max_length=args.max_length,
        )
        test_dataset = UnifiedPeptideDataset(
            sequences=seq_test, charges=np.asarray(ch_test), targets=np.asarray(tgt_test),
            max_length=args.max_length, aux_features=aux_test,
            target_mean=target_mean, target_std=target_std, normalize_target=True,
        )
        test_set = test_dataset
    else:
        test_set = torch.utils.data.Subset(full_dataset, test_idx)

    use_cuda = torch.cuda.is_available()
    loader_kw = {
        "batch_size": args.batch_size,
        "collate_fn": collate_unified_batch,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda and args.num_workers > 0,
        "persistent_workers": args.num_workers > 0,
    }
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, **loader_kw
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        **loader_kw,
    )
    vocab_size = get_vocab_size()
    pad_idx = get_pad_idx()
    num_charges = max(10, int(charges.max()))

    model = UnifiedRNNCCSPredictor(
        vocab_size=vocab_size,
        max_sequence_length=args.max_length,
        embed_dim=64,
        hidden_dim=256,
        num_layers=2,
        num_charges=num_charges,
        charge_embed_dim=32,
        aux_dim=aux_dim,
        aux_encoder_dim=32,
        dropout=0.3,
        padding_idx=pad_idx,
        output_dim=1,
    )

    trainer = UnifiedTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=1e-5,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=args.amp,
    )
    save_path = os.path.join(output_dir, "best_unified_model.pt")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=save_path,
        loss_type=args.loss,
        val_every=args.val_every,
    )

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best for evaluation
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
    device = trainer.device
    model.to(device)

    # Test set evaluation
    preds, targets, charges_arr, seq_lengths, cos_sim = evaluate_model(
        model, test_loader, device, target_mean, target_std
    )
    per_ch = charge_stratified_metrics(preds, targets, charges_arr)
    # Save with 1-based charge keys (z=1,2,3) for readability
    per_ch_display = {str(int(k) + 1): v for k, v in per_ch.items()}
    with open(os.path.join(output_dir, "charge_stratified_metrics.json"), "w") as f:
        json.dump(per_ch_display, f, indent=2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print("\nUnified model – test set:")
    print(f"  RMSE = {np.sqrt(mean_squared_error(targets, preds)):.4f}")
    print(f"  MAE  = {mean_absolute_error(targets, preds):.4f}")
    print(f"  R²   = {r2_score(targets, preds):.4f}")
    print("Per charge (z=1,2,3):", per_ch_display)

    plot_charge_stratified(
        preds, targets, charges_arr, seq_lengths, cos_sim, output_dir
    )

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="Train")
    ax[0].plot(history["val_loss"], label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_title("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["train_r2"], label="Train")
    ax[1].plot(history["val_r2"], label="Val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("R²")
    ax[1].legend()
    ax[1].set_title("R²")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unified_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Results and plots saved to {output_dir}")


if __name__ == "__main__":
    main()
