#!/usr/bin/env python3
"""
Unified Feature Embedding for Peptide CCS / Fragment Prediction
================================================================

This module provides:
1. Amino acid vocabulary and integer encoding (AAs → indices)
2. Sequence padding to max length L
3. Charge state encoding (scalar ∈ {1, 2, 3, …} → learned embedding index)
4. Optional auxiliary features (TIMS-specific): CCS, 1/K0, RT

Used by the unified RNN model for conditioning on charge and optional aux features.

Author: Unified CCS / Fragment Prediction
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


# Standard amino acid alphabet (20 canonical + padding)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
# Vocab: PAD=0, UNK=1, then A,C,D,... (indices 2..21)
VOCAB = [PAD_TOKEN, UNK_TOKEN] + list(AA_ALPHABET)
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA: Dict[int, str] = {i: aa for aa, i in AA_TO_IDX.items()}
PAD_IDX = AA_TO_IDX[PAD_TOKEN]
UNK_IDX = AA_TO_IDX[UNK_TOKEN]


def sequence_to_indices(sequence: str, max_length: int) -> np.ndarray:
    """
    Encode a peptide sequence to integer indices with padding.

    Args:
        sequence: Amino acid sequence (e.g. "MKFLVN")
        max_length: Max length L; shorter sequences are right-padded with PAD_IDX.

    Returns:
        int array of shape (max_length,) with values in [0, vocab_size-1].
    """
    seq_upper = sequence.upper().strip()
    indices = []
    for c in seq_upper:
        indices.append(AA_TO_IDX.get(c, UNK_IDX))
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [PAD_IDX] * (max_length - len(indices))
    return np.array(indices, dtype=np.int64)


def get_vocab_size() -> int:
    """Return vocabulary size (including PAD and UNK)."""
    return len(VOCAB)


def get_pad_idx() -> int:
    return PAD_IDX


def get_unk_idx() -> int:
    return UNK_IDX


def charge_to_index(charge: int, max_charge: int = 10) -> int:
    """
    Map charge state to a 0-based index for embedding lookup.
    charge in {1, 2, 3, ...} -> index in {0, 1, 2, ...}.
    Clamp to [0, max_charge-1] for embedding table size.
    """
    idx = max(0, int(charge) - 1)
    return min(idx, max_charge - 1)


def build_aux_features(
    row: pd.Series,
    aux_columns: Optional[List[str]] = None,
    fill_missing: float = 0.0,
) -> np.ndarray:
    """
    Build optional auxiliary feature vector from a data row.

    Typical columns: CCS, 1/K0, RT (or predicted RT).
    Missing values are filled with fill_missing.

    Args:
        row: pandas Series (one row of dataframe)
        aux_columns: List of column names, e.g. ['CCS_Experimental', 'inv_K0', 'RT']
        fill_missing: Value to use when column is missing or NaN

    Returns:
        float array of shape (len(aux_columns),).
    """
    if aux_columns is None:
        return np.array([], dtype=np.float32)
    out = []
    for col in aux_columns:
        if col in row.index and pd.notna(row[col]):
            try:
                out.append(float(row[col]))
            except (TypeError, ValueError):
                out.append(fill_missing)
        else:
            out.append(fill_missing)
    return np.array(out, dtype=np.float32)


class UnifiedPeptideDataset(torch.utils.data.Dataset):
    """
    Dataset that yields:
        sequence: int tensor (L,)
        charge: int (0-based index for embedding)
        aux: float tensor (n_aux,) or empty
        target: float (CCS or single regression target)
    Optional: sequence_length (int) for masking / analysis.
    """

    def __init__(
        self,
        sequences: List[str],
        charges: Union[List[int], np.ndarray],
        targets: Union[List[float], np.ndarray],
        max_length: int,
        aux_features: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        normalize_target: bool = True,
    ):
        """
        Args:
            sequences: List of peptide sequences
            charges: Charge state per sample (1, 2, 3, ...)
            targets: Regression target (e.g. CCS)
            max_length: Max sequence length L
            aux_features: Optional (N, n_aux) array; if None, aux is zeros
            target_mean, target_std: For normalizing target (computed from train if None)
            normalize_target: Whether to normalize target
        """
        self.sequences = list(sequences)
        self.charges = np.asarray(charges, dtype=np.int64)
        self.targets_raw = np.asarray(targets, dtype=np.float32)
        self.max_length = max_length

        # Pre-compute sequence indices and lengths for fast __getitem__
        n = len(self.sequences)
        self._seq_indices = np.zeros((n, max_length), dtype=np.int64)
        self._seq_lengths = np.zeros(n, dtype=np.int64)
        for i in range(n):
            self._seq_indices[i] = sequence_to_indices(self.sequences[i], max_length)
            self._seq_lengths[i] = min(len(self.sequences[i].strip()), max_length)

        if aux_features is not None:
            self.aux_features = np.asarray(aux_features, dtype=np.float32)
            assert len(self.aux_features) == len(self.sequences)
        else:
            self.aux_features = np.zeros((len(self.sequences), 0), dtype=np.float32)

        if normalize_target:
            if target_mean is None:
                target_mean = float(np.mean(self.targets_raw))
            if target_std is None:
                target_std = float(np.std(self.targets_raw))
            if target_std < 1e-8:
                target_std = 1.0
            self.target_mean = target_mean
            self.target_std = target_std
            self.targets = (self.targets_raw - self.target_mean) / self.target_std
        else:
            self.target_mean = 0.0
            self.target_std = 1.0
            self.targets = self.targets_raw.copy()

        self.normalize_target = normalize_target

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, float, int]:
        seq_idx = torch.from_numpy(self._seq_indices[idx])
        charge_idx = charge_to_index(int(self.charges[idx]))
        aux = torch.as_tensor(self.aux_features[idx], dtype=torch.float32)
        target = float(self.targets[idx])
        seq_len = int(self._seq_lengths[idx])
        return (seq_idx, charge_idx, aux, target, seq_len)

    def get_normalization_params(self) -> Dict[str, float]:
        return {"target_mean": self.target_mean, "target_std": self.target_std}


def load_tsv_for_unified(
    path: str,
    sequence_column: str = "Sequence",
    charge_column: str = "Charge",
    target_column: str = "CCS_Experimental",
    aux_columns: Optional[List[str]] = None,
    max_length: int = 50,
    sep: str = "\t",
) -> Tuple[List[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a TSV and return lists/arrays for UnifiedPeptideDataset.

    Returns:
        sequences, charges, targets, aux_features (or None)
    """
    df = pd.read_csv(path, sep=sep)
    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not in {list(df.columns)}")
    if charge_column not in df.columns:
        raise ValueError(f"Charge column '{charge_column}' not in {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in {list(df.columns)}")

    sequences = df[sequence_column].astype(str).str.strip().tolist()
    charges = df[charge_column].values.astype(np.int64)
    targets = df[target_column].values.astype(np.float32)

    aux_features = None
    if aux_columns:
        present = [c for c in aux_columns if c in df.columns]
        if present:
            aux_features = np.column_stack([
                df[c].fillna(0.0).astype(np.float32).values for c in present
            ])
        else:
            aux_features = np.zeros((len(sequences), len(aux_columns)), dtype=np.float32)

    return sequences, charges, targets, aux_features


def stratified_split_indices_by_charge(
    charges: Union[List[int], np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return train/val/test indices so that each charge appears proportionally in each split.
    Use these indices to compute train-only normalization, then build dataset and Subsets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    charges = np.asarray(charges)
    unique_charges = np.unique(charges)
    train_idx, val_idx, test_idx = [], [], []

    for ch in unique_charges:
        mask = charges == ch
        indices = np.where(mask)[0].tolist()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def stratified_split_by_charge(
    dataset: UnifiedPeptideDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Split dataset so that each charge appears in train/val/test proportionally.
    """
    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        dataset.charges, train_ratio, val_ratio, test_ratio, seed
    )
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def collate_unified_batch(
    batch: List[Tuple[torch.Tensor, int, torch.Tensor, float, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate (sequence, charge, aux, target, seq_len) into batched tensors.
    """
    sequences = torch.stack([b[0] for b in batch])
    charges = torch.tensor([b[1] for b in batch], dtype=torch.long)
    aux = torch.stack([b[2] for b in batch])
    targets = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    seq_lengths = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return sequences, charges, aux, targets, seq_lengths
