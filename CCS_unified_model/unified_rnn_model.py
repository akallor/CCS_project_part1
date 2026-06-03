#!/usr/bin/env python3
"""
Unified RNN (Sequence + Charge + Aux) Model for CCS / Fragment Prediction
===========================================================================

Single model trained on all charge states (1, 2, 3, …) with:
- Integer-encoded peptide sequence → AA embeddings → Sequence encoder (Bi-LSTM)
- Charge state → learned embedding
- Optional auxiliary features (CCS, 1/K0, RT) → small MLP
- Joint feature vector → MLP head → output (e.g. CCS or fragment intensities)

Architecture (concatenation of sequence + charge + aux, not addition).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class SequenceEncoder(nn.Module):
    """
    Sequence encoder: embed -> Bi-LSTM -> pooling (mean + max + last).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2 * 3  # mean, max, last for bidirectional

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)
        if self.embed.padding_idx is not None:
            self.embed.weight.data[self.embed.padding_idx].zero_()
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sequence: (B, L), lengths: (B,) optional.
        Returns: (B, output_dim) = concat(mean_pool, max_pool, last_hidden).
        """
        B, L = sequence.shape
        x = self.embed(sequence)  # (B, L, embed_dim)
        packed = x
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        out, (hn, _) = self.lstm(packed)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # mean pool (mask padding if needed)
        if lengths is not None:
            # pad_packed_sequence returns out with time dim = max(lengths), not L; mask must match out
            actual_L = out.size(1)
            mask = torch.arange(actual_L, device=sequence.device)[None, :] < lengths[:, None]
            mask = mask.float().unsqueeze(2)
            out_masked = out * mask
            mean_pool = out_masked.sum(1) / (lengths.float().unsqueeze(1).clamp(min=1))
        else:
            mean_pool = out.mean(1)
        max_pool = out.max(1)[0]
        last_hidden = out[:, -1, :]
        return torch.cat([mean_pool, max_pool, last_hidden], dim=1)


class ChargeEmbedding(nn.Module):
    """Maps charge index (0-based) to learned embedding vector."""

    def __init__(self, num_charges: int = 10, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_charges, embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)

    def forward(self, charge: torch.Tensor) -> torch.Tensor:
        """charge: (B,) long -> (B, embed_dim)."""
        return self.embed(charge)


class AuxiliaryEncoder(nn.Module):
    """Optional: encode auxiliary features (CCS, 1/K0, RT) with a small MLP."""

    def __init__(self, aux_dim: int, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()
        if aux_dim == 0:
            self.mlp = None
            self.output_dim = 0
            return
        self.mlp = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        """aux: (B, aux_dim) -> (B, output_dim) or (B, 0)."""
        if self.mlp is None:
            return aux.new_empty(aux.size(0), 0)
        return self.mlp(aux)


class UnifiedRNNCCSPredictor(nn.Module):
    """
    Unified model: sequence encoder + charge embedding + optional aux encoder
    -> concat -> MLP head -> single output (e.g. CCS).
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_charges: int = 10,
        charge_embed_dim: int = 32,
        aux_dim: int = 0,
        aux_encoder_dim: int = 32,
        dropout: float = 0.3,
        padding_idx: int = 0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.seq_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.charge_embed = ChargeEmbedding(num_charges=num_charges, embed_dim=charge_embed_dim)
        self.aux_encoder = AuxiliaryEncoder(
            aux_dim=aux_dim,
            hidden_dim=min(aux_dim, 64) if aux_dim else 0,
            output_dim=aux_encoder_dim,
        )

        joint_dim = (
            self.seq_encoder.output_dim
            + charge_embed_dim
            + self.aux_encoder.output_dim
        )
        self.head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.LayerNorm(joint_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim // 2, joint_dim // 4),
            nn.LayerNorm(joint_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim // 4, output_dim),
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
        sequence: torch.Tensor,
        charge: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sequence: (B, L), charge: (B,) long, aux: (B, aux_dim) or None.
        Returns: (B, output_dim).
        """
        h_seq = self.seq_encoder(sequence, lengths=lengths)
        h_charge = self.charge_embed(charge)
        if aux is not None and aux.size(1) > 0:
            h_aux = self.aux_encoder(aux)
            joint = torch.cat([h_seq, h_charge, h_aux], dim=1)
        else:
            joint = torch.cat([h_seq, h_charge], dim=1)
        return self.head(joint)
