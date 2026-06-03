#!/usr/bin/env python3
"""
Unified ESM-2 + RNN CCS Predictor
=================================

Unified CCS predictor that combines:
1. ESM-2 pooled embeddings (from charge_aware_esm_feature_extraction)
2. Chemical/engineered features for charge (nK, nR, nH, nD, nE, length, net_basicity)
3. Ground-truth charge as learned embedding (like unified predictor)
4. Simple 2-layer RNN (LSTM or GRU) on the concatenated representation
5. MLP head → single CCS output

This is the ESM-based analogue of UnifiedRNNCCSPredictor: same unified
training setup (all charges, stratified split) but with ESM-2 features
instead of learned amino-acid embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# Engineered feature dimension from charge_aware_esm_feature_extraction
ENGINEERED_FEAT_DIM = 7  # nK, nR, nH, nD, nE, length, net_basicity


class UnifiedESMCCSPredictor(nn.Module):
    """
    Unified ESM-2 + RNN CCS predictor.

    Inputs: pooled ESM (B, esm_dim), engineered features (B, 7), charge (B,) 1-based.
    Concatenates with charge embedding → projects to RNN input → 2-layer RNN →
    MLP head → CCS (B, 1).
    """

    def __init__(
        self,
        esm_dim: int,
        engineered_feat_dim: int = ENGINEERED_FEAT_DIM,
        num_charges: int = 10,
        charge_embed_dim: int = 32,
        rnn_input_dim: int = 256,
        rnn_hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.engineered_feat_dim = engineered_feat_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        joint_input_dim = esm_dim + engineered_feat_dim + charge_embed_dim

        self.charge_embed = nn.Embedding(num_charges, charge_embed_dim)
        nn.init.normal_(self.charge_embed.weight, mean=0.0, std=0.1)

        # Project joint vector to RNN input (single "sequence" of length 1 or a few steps)
        self.joint_to_rnn = nn.Sequential(
            nn.Linear(joint_input_dim, rnn_input_dim),
            nn.LayerNorm(rnn_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_dim,
                rnn_hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                rnn_input_dim,
                rnn_hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.RNN(
                rnn_input_dim,
                rnn_hidden_dim,
                num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=dropout if num_layers > 1 else 0,
            )

        rnn_out_dim = rnn_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim // 2),
            nn.LayerNorm(rnn_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim // 2, output_dim),
        )
        self._init_head()

    def _init_head(self):
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
        """
        pooled_esm: (B, esm_dim)
        engineered: (B, engineered_feat_dim)
        charge: (B,) long, 0-based (use charge - 1 when passing from dataset)
        Returns: (B, output_dim)
        """
        h_charge = self.charge_embed(charge)
        joint = torch.cat([pooled_esm, engineered, h_charge], dim=1)
        rnn_in = self.joint_to_rnn(joint)
        rnn_in = rnn_in.unsqueeze(1)
        out, _ = self.rnn(rnn_in)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)
