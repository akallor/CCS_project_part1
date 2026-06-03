#!/usr/bin/env python3
"""
Unified ESM-2 + RNN CCS Training – Charge 1 Only
================================================

Trains the same Unified ESM-2 + RNN architecture as run_unified_esm_training_corrected.py,
but only on peptides with Charge = 1 (z=1).

Why use this script?
  The unified (all-charge) model is trained on a mix of z=1, z=2, z=3, etc. When the
  dataset is imbalanced (e.g. many more z=2 than z=1), the shared model tends to fit
  the majority charges better and can underfit z=1, giving lower R² and systematic
  residuals (e.g. over-prediction at low CCS, under-prediction at high CCS for z=1).
  Training a charge-1-only model dedicates all capacity to z=1 and typically improves
  R² and residual behavior for that charge.

Same as corrected script:
  - Eval-mode train/val metrics, train-only CCS normalization, stratified split
    (here a single charge so 80/10/10 random split), same model architecture
    (num_charges=1), same plots and outputs.

Defaults aligned with rnn_charge_aware_ccs_predictor_improved.py for better R²:
  - lr=5e-4, dropout=0.4, loss=mse, weight_decay=1e-4, patience=10, min_delta=5e-4.
  - ReduceLROnPlateau scheduler (factor=0.5, patience=5) to refine when val plateaus.

To target R² ~0.85+ (comparable to other charges):
  - --use_enhanced_features: use 17-dim physics-informed features from TSV (Peptide or Sequence column required).
  - --improved_arch: use larger bidirectional model (512/256) like improved script; auto-enables
    enhanced features. Use both for best charge-1 performance.
  - --mlp_trunk: for improved_arch only, replace the 1-step BiLSTM with a residual MLP trunk
    (often better for pooled ESM—there is no sequence for the LSTM to exploit).
  - --log1p_ccs_target: optimize in log1p(CCS) space (z-scored); metrics stay in Å².
  If R² stays ~0.62–0.66 with t6_8M + mean pool, try re-extracting with a larger ESM
  (e.g. esm2_t33_650M_UR50D in run_extract_esm_for_unified.py).

Scale-aware loss (Step 2; z=1 CCS range is narrow vs higher charges in unified training):
  - --loss relative_mse: MSE in raw CCS divided by (CCS² + eps²) so gradients are not dominated by large-CCS scale.
  - --loss mape: mean |error| / (|CCS| + eps).
  - --loss norm_huber: Huber on error / (|CCS| + eps).

Architecture (charge-1-only):
  - --charge_aware_steps N (N>0): LSTM over N timesteps; each step gets ESM+eng backbone + charge embed + step embedding.
  - --z1_deep_head: deeper MLP head (512→256→128→1) for z=1-specific mapping.

Usage:
  1. Extract features for the full dataset (same as unified): run_extract_esm_for_unified.py
  2. Train charge-1-only (base): ... --data_path your_data.tsv --features_path your_features.pt
  3. Train for best R² (enhanced + improved arch): add --use_enhanced_features --improved_arch
     (TSV must contain Peptide or Sequence; optional Mass column for 17-dim features)
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

try:
    from unified_esm_rnn_model import UnifiedESMCCSPredictorLarge
except ImportError:
    # Define locally if not in unified_esm_rnn_model (e.g. older HPC copy)
    class UnifiedESMCCSPredictorLarge(nn.Module):
        """Larger ESM-2 + RNN predictor (bidirectional 512/256). Same API as UnifiedESMCCSPredictor."""

        def __init__(
            self,
            esm_dim: int,
            engineered_feat_dim: int = 17,
            num_charges: int = 10,
            charge_embed_dim: int = 32,
            joint_dim: int = 512,
            rnn_hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
            output_dim: int = 1,
            deep_z1_head: bool = False,
            mlp_trunk: bool = False,
            mlp_trunk_blocks: int = 4,
        ):
            super().__init__()
            self.mlp_trunk = mlp_trunk
            joint_input_dim = esm_dim + engineered_feat_dim + charge_embed_dim
            self.charge_embed = nn.Embedding(num_charges, charge_embed_dim)
            nn.init.normal_(self.charge_embed.weight, mean=0.0, std=0.1)
            self.joint_to_rnn = nn.Sequential(
                nn.Linear(joint_input_dim, joint_dim),
                nn.LayerNorm(joint_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if mlp_trunk:
                self.rnn = None
                self.trunk_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(joint_dim, joint_dim),
                            nn.LayerNorm(joint_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )
                        for _ in range(max(1, int(mlp_trunk_blocks)))
                    ]
                )
                rnn_out_dim = joint_dim
            else:
                self.trunk_blocks = None
                self.rnn = nn.LSTM(
                    joint_dim, rnn_hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
                )
                rnn_out_dim = rnn_hidden_dim * 2
            if deep_z1_head:
                self.head = nn.Sequential(
                    nn.Linear(rnn_out_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, output_dim),
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(rnn_out_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, output_dim),
                )
            for m in self.head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        def forward(self, pooled_esm: torch.Tensor, engineered: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
            h_charge = self.charge_embed(charge)
            joint = torch.cat([pooled_esm, engineered, h_charge], dim=1)
            x = self.joint_to_rnn(joint)
            if self.mlp_trunk:
                for blk in self.trunk_blocks:
                    x = x + blk(x)
                return self.head(x)
            rnn_in = x.unsqueeze(1)
            out, (h_n, _) = self.rnn(rnn_in)
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return self.head(last_hidden)


class UnifiedESMCCSPredictorChargeEveryStep(nn.Module):
    """
    Charge embedding concatenated at every LSTM timestep with backbone + learnable step index.
    Backbone: ESM+eng → MLP → h; each step t: [h || charge_emb || step_emb_t] → proj → LSTM.
    """

    def __init__(
        self,
        esm_dim: int,
        engineered_feat_dim: int,
        num_charges: int = 1,
        charge_embed_dim: int = 16,
        num_steps: int = 4,
        step_dim: int = 8,
        lstm_input_dim: int = 512,
        rnn_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        deep_z1_head: bool = False,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.charge_embed = nn.Embedding(num_charges, charge_embed_dim)
        nn.init.normal_(self.charge_embed.weight, mean=0.0, std=0.1)
        self.step_emb = nn.Parameter(torch.randn(num_steps, step_dim) * 0.02)
        bb_in = esm_dim + engineered_feat_dim
        self.backbone = nn.Sequential(
            nn.Linear(bb_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        per_step_in = 512 + charge_embed_dim + step_dim
        self.per_step_proj = nn.Linear(per_step_in, lstm_input_dim)
        self.rnn = nn.LSTM(
            lstm_input_dim,
            rnn_hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        out_dim = rnn_hidden_dim * 2
        if deep_z1_head:
            self.head = nn.Sequential(
                nn.Linear(out_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(out_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        pooled_esm: torch.Tensor,
        engineered: torch.Tensor,
        charge: torch.Tensor,
    ) -> torch.Tensor:
        h = self.backbone(torch.cat([pooled_esm, engineered], dim=1))
        c = self.charge_embed(charge)
        B = h.size(0)
        steps = self.step_emb.unsqueeze(0).expand(B, -1, -1)
        h_rep = h.unsqueeze(1).expand(-1, self.num_steps, -1)
        c_rep = c.unsqueeze(1).expand(-1, self.num_steps, -1)
        seq_in = torch.cat([h_rep, c_rep, steps], dim=-1)
        x = self.per_step_proj(seq_in)
        _, (h_n, _) = self.rnn(x)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.head(last_hidden)


def make_charge1_criterion(
    loss_type: str,
    target_mean: float,
    target_std: float,
    eps_ccs: float = 50.0,
    target_is_log1p: bool = False,
):
    """
    Loss on normalized pred/target; relative_mse/mape/norm_huber use raw CCS for scaling.
    If target_is_log1p, normalized targets are z-scored log1p(CCS); to_raw maps to Å² via expm1.
    """
    tm = float(target_mean)
    ts = float(target_std)

    def to_raw(p, t):
        pn = p.squeeze(-1) * ts + tm
        tn = t.squeeze(-1) * ts + tm
        if target_is_log1p:
            return torch.expm1(pn), torch.expm1(tn)
        return pn, tn

    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "huber":
        return nn.HuberLoss()

    class _RelMSE(nn.Module):
        def forward(self, pred, target):
            pr, tr = to_raw(pred, target)
            return torch.mean((pr - tr) ** 2 / (tr * tr + eps_ccs * eps_ccs))

    class _MAPE(nn.Module):
        def forward(self, pred, target):
            pr, tr = to_raw(pred, target)
            return torch.mean(torch.abs(pr - tr) / (torch.abs(tr) + eps_ccs))

    class _NormHuber(nn.Module):
        def forward(self, pred, target):
            pr, tr = to_raw(pred, target)
            rel = (pr - tr) / (torch.abs(tr) + eps_ccs)
            return torch.nn.functional.smooth_l1_loss(rel, torch.zeros_like(rel))

    if loss_type == "relative_mse":
        return _RelMSE()
    if loss_type == "mape":
        return _MAPE()
    if loss_type == "norm_huber":
        return _NormHuber()
    raise ValueError(f"Unknown loss_type: {loss_type}")


# ---------------------------------------------------------------------------
# 17-dim enhanced features (from rnn_charge_aware_ccs_predictor_improved.py)
# ---------------------------------------------------------------------------

HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE_AA = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}
AVG_RESIDUE_MASS = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2, "G": 75.1, "H": 155.2,
    "I": 131.2, "K": 146.2, "L": 131.2, "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2,
    "R": 174.2, "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}
DEFAULT_AVG_MASS = 110.0


def _mass_from_sequence(sequence: str) -> float:
    total = 0.0
    for aa in sequence.upper():
        total += AVG_RESIDUE_MASS.get(aa, DEFAULT_AVG_MASS)
    return total


def calculate_enhanced_features_17(sequence: str, mass: float, charge_state: int = 1) -> np.ndarray:
    """17 physics-informed features (from improved script). charge_state=1 for charge-1-only."""
    length = max(len(sequence), 1)
    features = {"length": length, "mass": mass, "charge_state": charge_state}
    hydrophobicity_values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    features["hydrophobicity_mean"] = float(np.mean(hydrophobicity_values)) if hydrophobicity_values else 0.0
    features["hydrophobicity_std"] = float(np.std(hydrophobicity_values)) if len(hydrophobicity_values) > 1 else 0.0
    charge_values = [CHARGE_AA.get(aa, 0) for aa in sequence]
    charged_positions = [i for i, c in enumerate(charge_values) if c != 0]
    if len(charged_positions) >= 2:
        spacings = [charged_positions[i + 1] - charged_positions[i] for i in range(len(charged_positions) - 1)]
        features["charge_spacing"] = float(np.mean(spacings))
        center_of_charge = np.mean(charged_positions)
        features["charge_asymmetry"] = abs(center_of_charge - length / 2) / (length / 2)
    else:
        features["charge_spacing"] = 0.0
        features["charge_asymmetry"] = 0.0
    features["aromatic_count"] = sum(1 for aa in sequence if aa in "FWY")
    features["proline_count"] = sequence.count("P")
    features["glycine_count"] = sequence.count("G")
    features["basic_count"] = sum(1 for aa in sequence if aa in "KRH")
    features["acidic_count"] = sum(1 for aa in sequence if aa in "DE")
    features["flexibility"] = (sequence.count("G") + sequence.count("P")) / length
    features["estimated_rg"] = 2.2 * (length ** 0.38)
    features["projected_area_estimate"] = mass ** (2 / 3)
    features["charge_per_residue"] = 1.0 / length
    features["mass_per_charge"] = mass
    return np.array([
        features["length"], features["mass"], features["charge_state"],
        features["hydrophobicity_mean"], features["hydrophobicity_std"],
        features["charge_spacing"], features["charge_asymmetry"],
        features["aromatic_count"], features["proline_count"], features["glycine_count"],
        features["basic_count"], features["acidic_count"], features["flexibility"],
        features["estimated_rg"], features["projected_area_estimate"],
        features["charge_per_residue"], features["mass_per_charge"],
    ], dtype=np.float32)


class FeatureNormalizer:
    """Fit on train engineered features only; transform train/val/test."""

    def __init__(self) -> None:
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("FeatureNormalizer must be fit before transform.")
        return (features - self.means) / self.stds


def build_enhanced_features_charge1(
    data_path: str,
    n_total: int,
    idx: np.ndarray,
    train_idx: List[int],
    charge_val: int = 1,
) -> Tuple[torch.Tensor, "FeatureNormalizer"]:
    """
    Load TSV, take rows at indices idx (charge-1 subset), compute 17-dim features.
    Fit FeatureNormalizer on train_idx portion; return (normalized tensor, normalizer).
    """
    df = pd.read_csv(data_path, sep="\t")
    df = df.iloc[:n_total]
    df_charge1 = df.iloc[idx].reset_index(drop=True)
    seq_col = None
    for c in ["Peptide", "Sequence", "Modified Sequence", "sequence", "peptide"]:
        if c in df_charge1.columns:
            seq_col = c
            break
    if seq_col is None:
        raise ValueError("TSV must have a sequence column (e.g. Peptide, Sequence) for --use_enhanced_features.")
    mass_col = None
    for c in ["Mass", "Molecular Weight", "Precursor Mass", "mass"]:
        if c in df_charge1.columns:
            mass_col = c
            break
    rows = []
    for i in range(len(df_charge1)):
        seq = str(df_charge1.iloc[i][seq_col]).strip().upper()
        if mass_col is not None:
            try:
                mass = float(df_charge1.iloc[i][mass_col])
            except Exception:
                mass = _mass_from_sequence(seq)
        else:
            mass = _mass_from_sequence(seq)
        rows.append(calculate_enhanced_features_17(seq, mass, charge_val))
    engineered_17 = np.stack(rows)
    normalizer = FeatureNormalizer()
    normalizer.fit(engineered_17[train_idx])
    engineered_17_norm = normalizer.transform(engineered_17)
    return torch.tensor(engineered_17_norm.astype(np.float32)), normalizer


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
        use_lr_scheduler: bool = True,
        loss_eps_ccs: float = 50.0,
        target_is_log1p: bool = False,
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
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )
        else:
            self.scheduler = None
        self.loss_eps_ccs = loss_eps_ccs
        self.target_is_log1p = target_is_log1p
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [],
            "train_r2": [], "val_r2": [], "lr": [],
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

    @torch.no_grad()
    def evaluate(
        self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module
    ) -> Tuple[float, float, float, float]:
        """Returns (criterion_loss, mse_normalized, r2_on_ccs, rmse_ccs_angstrom)."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_r2 = 0.0
        sse_ccs = 0.0
        n_ccs = 0
        n_batches = 0
        non_blocking = self.device.type == "cuda"
        tm, ts = self.target_mean, self.target_std

        for esm, eng, charge, target in data_loader:
            esm = esm.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            pred = self.model(pooled_esm=esm, engineered=eng, charge=charge).squeeze(-1)
            loss = criterion(pred, target)
            mse = ((pred - target) ** 2).mean().item()
            if self.target_is_log1p:
                pred_ccs = torch.expm1(pred * ts + tm)
                target_ccs = torch.expm1(target * ts + tm)
            else:
                pred_ccs = pred * ts + tm
                target_ccs = target * ts + tm
            r2 = _r2_torch(pred_ccs, target_ccs)
            sse_ccs += ((pred_ccs - target_ccs) ** 2).sum().item()
            n_ccs += pred_ccs.numel()
            total_loss += loss.item()
            total_mse += mse
            total_r2 += r2
            n_batches += 1

        n = max(n_batches, 1)
        rmse_ccs = float(np.sqrt(sse_ccs / max(n_ccs, 1)))
        return total_loss / n, total_mse / n, total_r2 / n, rmse_ccs

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
    ) -> Dict[str, List[float]]:
        criterion = make_charge1_criterion(
            loss_type,
            self.target_mean,
            self.target_std,
            eps_ccs=self.loss_eps_ccs,
            target_is_log1p=self.target_is_log1p,
        )
        criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(num_epochs):
            self.train_epoch(train_loader, criterion, epoch, num_epochs)

            # Report train and val metrics in eval mode for fair comparison (no dropout)
            train_loss, train_mse, train_r2, train_rmse_ccs = self.evaluate(
                train_loader, criterion
            )
            do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
            if do_val:
                val_loss, val_mse, val_r2, val_rmse_ccs = self.evaluate(
                    val_loader, criterion
                )
            else:
                val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else float("inf")
                val_mse = self.history["val_mse"][-1] if self.history["val_mse"] else float("inf")
                val_r2 = self.history["val_r2"][-1] if self.history["val_r2"] else 0.0
                val_rmse_ccs = (
                    self.history["val_rmse_ccs"][-1]
                    if self.history.get("val_rmse_ccs")
                    else 0.0
                )

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)
            self.history["train_r2"].append(train_r2)
            self.history["val_r2"].append(val_r2)
            self.history["lr"].append(lr)
            self.history.setdefault("train_rmse_ccs", []).append(train_rmse_ccs)
            self.history.setdefault("val_rmse_ccs", []).append(val_rmse_ccs)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(
                    f"Epoch {epoch:3d}  train_RMSE_ccs={train_rmse_ccs:.4f}  val_RMSE_ccs={val_rmse_ccs:.4f}  "
                    f"train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  lr={lr:.2e}"
                )

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
    residuals = preds - targets

    # Combined plots (all charges; for charge-1-only this is the same as per-charge)
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

    # Per-charge plots (for charge-1-only, single charge)
    for ch in ch_list:
        mask = charges == ch
        t_ch = targets[mask]
        p_ch = preds[mask]
        r_ch = p_ch - t_ch
        z_label = int(ch) + 1
        n_ch = mask.sum()

        fig_ch, ax_ch = plt.subplots(1, 3, figsize=(14, 5))
        fig_ch.suptitle(f"Unified ESM (charge-1-only) – Charge z={z_label} (n={n_ch})", fontsize=12, fontweight="bold")

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
    """Dataset of (pooled_esm, engineered, charge_0based, ccs_normalized or log1p-normalized)."""

    def __init__(
        self,
        esm_tensor: torch.Tensor,
        engineered_tensor: torch.Tensor,
        charges: np.ndarray,
        targets: np.ndarray,
        target_mean: float,
        target_std: float,
        use_log1p_target: bool = False,
    ):
        assert len(esm_tensor) == len(engineered_tensor) == len(charges) == len(targets)
        self.esm = esm_tensor
        self.engineered = engineered_tensor
        self.charges = torch.tensor(charges, dtype=torch.long)
        self.targets_raw = torch.tensor(targets, dtype=torch.float32)
        self.target_mean = target_mean
        self.target_std = target_std
        self.use_log1p_target = use_log1p_target
        if use_log1p_target:
            raw = np.maximum(np.asarray(targets, dtype=np.float64), 1e-6)
            logt = np.log1p(raw)
            self.targets = torch.tensor(
                (logt - target_mean) / target_std, dtype=torch.float32
            )
        else:
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
    parser = argparse.ArgumentParser(description="Unified ESM-2 + RNN CCS training (charge 1 only)")
    parser.add_argument("--data_path", type=str, required=True, help="TSV with Charge, CCS_Experimental (same order as features)")
    parser.add_argument("--features_path", type=str, required=True, help=".pt from ESM + engineered feature extraction")
    parser.add_argument("--output_dir", type=str, default="./unified_esm_results_charge1_only")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=5e-4, help="Min val_loss improvement to count as improvement")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate (5e-4 aligned with improved script)")
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber", "relative_mse", "mape", "norm_huber"],
        help="Training loss; relative_mse/mape/norm_huber are scale-aware on raw CCS (see --loss_eps_ccs)",
    )
    parser.add_argument(
        "--loss_eps_ccs",
        type=float,
        default=50.0,
        help="Stabilizer (Å²) for relative_mse/mape/norm_huber denominators",
    )
    parser.add_argument(
        "--charge_aware_steps",
        type=int,
        default=0,
        help="If >0, use LSTM with charge+step embedding at each timestep (N steps). Overrides single-step improved arch.",
    )
    parser.add_argument(
        "--z1_deep_head",
        action="store_true",
        help="Deeper output MLP (256→128→1) for z=1; applies to Large or charge-aware-step model",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--no_lr_scheduler", action="store_true", help="Disable ReduceLROnPlateau (use warmup+cosine instead)")
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--esm_dim", type=int, default=None, help="Infer from .pt if not set")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate (0.4 aligned with improved for better R²)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--use_enhanced_features", action="store_true", help="Use 17-dim physics-informed features from TSV (requires Peptide or Sequence column); aims for R² ~0.85+")
    parser.add_argument("--improved_arch", action="store_true", help="Use larger bidirectional model (512/256) like improved script; use with --use_enhanced_features for best R²")
    parser.add_argument(
        "--mlp_trunk",
        action="store_true",
        help="With --improved_arch: residual MLP trunk instead of 1-step BiLSTM (recommended for pooled ESM)",
    )
    parser.add_argument("--mlp_trunk_blocks", type=int, default=4, help="Residual blocks when --mlp_trunk (default 4)")
    parser.add_argument("--split_seed", type=int, default=42, help="RNG seed for train/val/test split")
    parser.add_argument("--seed", type=int, default=42, help="PyTorch/NumPy seed for weight init and loaders")
    parser.add_argument(
        "--log1p_ccs_target",
        action="store_true",
        help="Train on z-scored log1p(CCS); metrics/RMSE in original Å². Often helps skewed CCS.",
    )
    args = parser.parse_args()
    if args.improved_arch and not args.use_enhanced_features:
        args.use_enhanced_features = True
        print("--improved_arch set: enabling --use_enhanced_features for 17-dim features.")
    if args.mlp_trunk and not args.improved_arch:
        args.improved_arch = True
        args.use_enhanced_features = True
        print("--mlp_trunk: enabling --improved_arch and --use_enhanced_features.")
    if args.charge_aware_steps > 0 and not args.use_enhanced_features:
        print("Note: --charge_aware_steps with default .pt engineered features (7-d). Add --use_enhanced_features for 17-d.")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return
    if not os.path.exists(args.features_path):
        print(f"Features not found: {args.features_path}. Run ESM feature extraction first.")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    esm_tensor, eng_tensor, charges, targets = load_features_and_tsv(
        args.features_path, args.data_path
    )
    # Restrict to charge 1 only (0-based index 0)
    mask = charges == 0
    n_before = len(charges)
    idx = np.where(mask)[0]
    esm_tensor = esm_tensor[idx]
    eng_tensor = eng_tensor[idx]
    charges = charges[idx]  # all 0
    targets = targets[idx]
    n_samples = len(esm_tensor)
    print(f"Charge-1-only: n={n_samples} samples (filtered from {n_before} total).")

    if n_samples < 10:
        print("Too few charge-1 samples. Exiting.")
        return

    esm_dim = esm_tensor.shape[1]
    eng_dim = eng_tensor.shape[1]

    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        charges,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.split_seed,
    )
    if args.use_enhanced_features:
        try:
            eng_tensor, _ = build_enhanced_features_charge1(
                args.data_path, n_before, idx, train_idx, charge_val=1
            )
            eng_dim = 17
            print("Using 17-dim enhanced features (train-normalized).")
        except Exception as e:
            print(f"Failed to build enhanced features: {e}. Falling back to .pt engineered features.")
    if args.esm_dim is not None and args.esm_dim != esm_dim:
        print(f"Warning: --esm_dim {args.esm_dim} != inferred {esm_dim}; using {esm_dim}")

    train_targets = targets[train_idx]
    if args.log1p_ccs_target:
        train_log = np.log1p(np.maximum(train_targets.astype(np.float64), 1e-6))
        target_mean = float(np.mean(train_log))
        target_std = float(np.std(train_log))
        if target_std < 1e-8:
            target_std = 1.0
        print("Target: z-scored log1p(CCS); R²/RMSE reported on original CCS scale.")
    else:
        target_mean = float(np.mean(train_targets))
        target_std = float(np.std(train_targets))
        if target_std < 1e-8:
            target_std = 1.0

    full_dataset = UnifiedESMDataset(
        esm_tensor,
        eng_tensor,
        charges,
        targets,
        target_mean=target_mean,
        target_std=target_std,
        use_log1p_target=args.log1p_ccs_target,
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

    num_charges = 1
    if args.charge_aware_steps > 0:
        model = UnifiedESMCCSPredictorChargeEveryStep(
            esm_dim=esm_dim,
            engineered_feat_dim=eng_dim,
            num_charges=num_charges,
            charge_embed_dim=16,
            num_steps=args.charge_aware_steps,
            step_dim=8,
            lstm_input_dim=512,
            rnn_hidden_dim=256,
            num_layers=2,
            dropout=args.dropout,
            deep_z1_head=args.z1_deep_head,
        )
        print(
            f"Using charge-at-every-step BiLSTM (steps={args.charge_aware_steps}, deep_z1_head={args.z1_deep_head})."
        )
    elif args.improved_arch:
        model = UnifiedESMCCSPredictorLarge(
            esm_dim=esm_dim,
            engineered_feat_dim=eng_dim,
            num_charges=num_charges,
            charge_embed_dim=32,
            joint_dim=512,
            rnn_hidden_dim=256,
            num_layers=2,
            dropout=args.dropout,
            output_dim=1,
            deep_z1_head=args.z1_deep_head,
            mlp_trunk=args.mlp_trunk,
            mlp_trunk_blocks=args.mlp_trunk_blocks,
        )
        trunk_s = f"mlp_trunk×{args.mlp_trunk_blocks}" if args.mlp_trunk else "BiLSTM 512/256"
        print(
            f"Using improved-arch ({trunk_s}, deep_z1_head={args.z1_deep_head})."
        )
    else:
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

    trainer = UnifiedESMTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=args.amp,
        use_lr_scheduler=not args.no_lr_scheduler,
        loss_eps_ccs=args.loss_eps_ccs,
        target_is_log1p=args.log1p_ccs_target,
    )

    save_path = os.path.join(args.output_dir, "best_unified_esm_model_charge1.pt")
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=save_path,
        loss_type=args.loss,
        val_every=args.val_every,
        min_delta=args.min_delta,
    )

    train_rmse_orig = (
        target_std * np.sqrt(np.maximum(np.array(history["train_mse"], dtype=np.float64), 0.0))
    ).tolist()
    val_rmse_orig = (
        target_std * np.sqrt(np.maximum(np.array(history["val_mse"], dtype=np.float64), 0.0))
    ).tolist()
    history_out = {
        **history,
        "train_rmse_original_scale": train_rmse_orig,
        "val_rmse_original_scale": val_rmse_orig,
        "train_rmse_ccs": history.get("train_rmse_ccs", []),
        "val_rmse_ccs": history.get("val_rmse_ccs", []),
        "target_std_used": target_std,
        "log1p_ccs_target": args.log1p_ccs_target,
        "loss": args.loss,
        "loss_eps_ccs": args.loss_eps_ccs,
        "charge_aware_steps": args.charge_aware_steps,
        "z1_deep_head": args.z1_deep_head,
        "mlp_trunk": args.mlp_trunk,
        "mlp_trunk_blocks": args.mlp_trunk_blocks,
        "split_seed": args.split_seed,
        "seed": args.seed,
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
            pn = out.cpu().numpy() * target_std + target_mean
            tn = target.cpu().numpy() * target_std + target_mean
            if args.log1p_ccs_target:
                preds_list.append(np.expm1(pn))
                targets_list.append(np.expm1(tn))
            else:
                preds_list.append(pn)
                targets_list.append(tn)
            charges_list.append(charge.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets_test = np.concatenate(targets_list)
    charges_test = np.concatenate(charges_list)

    per_ch = charge_stratified_metrics(preds, targets_test, charges_test)
    per_ch_display = {str(int(k) + 1): v for k, v in per_ch.items()}
    with open(os.path.join(args.output_dir, "charge_stratified_metrics.json"), "w") as f:
        json.dump(per_ch_display, f, indent=2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print("\nUnified ESM (charge-1-only) – test set:")
    print(f"  RMSE = {np.sqrt(mean_squared_error(targets_test, preds)):.4f}")
    print(f"  MAE  = {mean_absolute_error(targets_test, preds):.4f}")
    print(f"  R²   = {r2_score(targets_test, preds):.4f}")
    print("Per charge:", per_ch_display)

    plot_charge_stratified(preds, targets_test, charges_test, args.output_dir)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    tr_plot = history.get("train_rmse_ccs") or train_rmse_orig
    va_plot = history.get("val_rmse_ccs") or val_rmse_orig
    ax[0].plot(tr_plot, label="Train")
    ax[0].plot(va_plot, label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("RMSE (CCS Å²)")
    ax[0].legend()
    ax[0].set_title("Loss (RMSE, original scale)")
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

