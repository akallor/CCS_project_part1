#!/usr/bin/env python3
"""
Extract ESM-2 + engineered features for the unified ESM CCS predictor.
=======================================================================

Reads a TSV with columns Sequence, Charge, CCS_Experimental (or similar),
extracts ESM-2 pooled embeddings and chemical/engineered features (nK, nR, nH,
nD, nE, length, net_basicity) in the same row order as the TSV, and saves
a .pt file for use with run_unified_esm_training.py.

Usage:
  python run_extract_esm_for_unified.py --data_path your_data.tsv --output_path your_features.pt
  python run_extract_esm_for_unified.py --data_path your_data.tsv --output_path hla1_features.pt --min_length 8 --max_length 12
  python run_extract_esm_for_unified.py --data_path your_data.tsv --output_path hla2_features.pt --min_length 15

Then train with:
  python run_unified_esm_training.py --data_path your_data.tsv --features_path your_features.pt --output_dir ./unified_esm_results
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os

from charge_aware_esm_feature_extraction import ESMFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 + engineered features for unified ESM training")
    parser.add_argument("--data_path", type=str, required=True, help="TSV with at least a Sequence column")
    parser.add_argument("--output_path", type=str, required=True, help="Output .pt file path")
    parser.add_argument("--sequence_column", type=str, default="Peptide", help="Column name for peptide sequence")
    parser.add_argument("--model_type", type=str, default="esm2_t6_8M_UR50D", help="ESM-2 model variant")
    parser.add_argument("--aggregation", type=str, default="global_mean", choices=["global_mean", "global_max", "attention_weighted"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for ESM forward (smaller if OOM)")
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum peptide length to include (0 disables lower bound)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Maximum peptide length to include (0 disables upper bound)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}")

    df = pd.read_csv(args.data_path, sep="\t")
    if args.sequence_column not in df.columns:
        raise ValueError(f"Column '{args.sequence_column}' not in {list(df.columns)}")

    sequences_all = df[args.sequence_column].astype(str).str.strip().tolist()
    lengths_all = np.array([len(s) for s in sequences_all], dtype=np.int64)
    keep_mask = np.ones(len(sequences_all), dtype=bool)
    if args.min_length > 0:
        keep_mask &= lengths_all >= args.min_length
    if args.max_length > 0:
        keep_mask &= lengths_all <= args.max_length
    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices) == 0:
        raise RuntimeError(
            f"No sequences left after length filtering with min={args.min_length}, max={args.max_length}"
        )
    sequences = [sequences_all[i] for i in kept_indices]
    print(
        f"Length filter: min={args.min_length}, max={args.max_length}; "
        f"kept {len(sequences)}/{len(sequences_all)} rows"
    )

    valid = []
    for i, s in enumerate(sequences):
        if s and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in s.upper()):
            valid.append((f"row_{i}", s.upper()))
        else:
            if s:
                print(f"Warning: skipping row {i} (invalid or empty sequence)")
            valid.append((f"row_{i}", "M"))
    batch_sequences = valid

    extractor = ESMFeatureExtractor(
        model_type=args.model_type,
        aggregation_strategy=args.aggregation,
        batch_size=args.batch_size,
    )
    extractor.load_esm_model()

    all_esm = []
    all_eng = []
    n = len(batch_sequences)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = batch_sequences[start:end]
        esm_batch, eng_batch = extractor.process_batch(batch)
        all_esm.extend(esm_batch)
        all_eng.extend(eng_batch)
        if (end - start) < args.batch_size or end % 100 == 0 or end == n:
            print(f"Processed {end}/{n} sequences")

    if not all_esm:
        raise RuntimeError("No features extracted")

    engineered_array = np.array(all_eng, dtype=np.float32)
    mean = engineered_array.mean(axis=0)
    std = engineered_array.std(axis=0) + 1e-8
    normalized_eng = (engineered_array - mean) / std
    normalization_params = {"mean": mean.tolist(), "std": std.tolist()}

    if isinstance(all_esm[0], torch.Tensor):
        esm_stack = torch.stack([t.cpu() for t in all_esm])
    else:
        esm_stack = torch.stack(all_esm)

    results = {
        "esm_features": esm_stack,
        "engineered_features": normalized_eng,
        "normalization_params": normalization_params,
        "selected_indices": kept_indices.tolist(),
        "length_filter": {
            "min_length": int(args.min_length),
            "max_length": int(args.max_length),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    torch.save(results, args.output_path)
    print(f"Saved {len(esm_stack)} samples to {args.output_path}")
    print(f"ESM dim: {esm_stack.shape[1]}, Engineered dim: {normalized_eng.shape[1]}")


if __name__ == "__main__":
    main()

