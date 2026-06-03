"""
ESM-2 Feature Extraction with Engineered Residue Features
==========================================================

This module provides feature extraction for peptide sequences using ESM-2 models
without explicit charge tokens. Instead, it:
1. Extracts ESM-2 embeddings (frozen model)
2. Computes engineered residue features (nK, nR, nH, nD, nE, length, net_basicity)
3. Returns both pooled ESM embeddings and engineered features

Key Features:
- ESM-2 embeddings (mean pooling, attention mask-aware)
- Engineered residue features for charge prediction
- No explicit charge tokens

Author: Re-engineered CCS Prediction System
Purpose: Extract features for hybrid ESM-2 + RNN CCS prediction
"""

import torch
import torch.nn as nn
import csv
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import warnings
from collections import defaultdict

class ESMFeatureExtractor:
    """
    ESM-2 feature extractor for peptide sequences.
    
    This class extracts ESM-2 embeddings without charge tokens and computes
    engineered residue features that help predict charge from sequence.
    """
    
    def __init__(self, 
                 model_type: str = "esm2_t6_8M_UR50D",
                 aggregation_strategy: str = "global_mean",
                 batch_size: int = 20000):
        """
        Initialize the ESM feature extractor.
        
        Args:
            model_type: ESM-2 model variant to use
            aggregation_strategy: Method for aggregating sequence features
            batch_size: Number of sequences to process per batch
        """
        self.model_variant = model_type
        self.aggregation_method = aggregation_strategy
        self.processing_batch_size = batch_size
        
        # Available ESM-2 models
        self.available_models = {
            "esm2_t6_8M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
            "esm2_t12_35M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
            "esm2_t30_150M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
            "esm2_t33_650M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
            "esm2_t36_3B_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
            "esm2_t48_15B_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt"
        }
        
        # Available aggregation strategies
        self.aggregation_strategies = [
            "global_mean", "global_max", "attention_weighted"
        ]
        
        self.esm_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics tracking
        self.extraction_stats = {
            "total_sequences": 0,
            "sequence_lengths": []
        }
    
    def load_esm_model(self) -> None:
        """
        Load the specified ESM-2 model and tokenizer.
        """
        print(f"Loading ESM-2 model: {self.model_variant}")
        
        try:
            # Load ESM-2 model using torch.hub
            self.esm_model, self.tokenizer = torch.hub.load(
                "facebookresearch/esm:main", 
                self.model_variant
            )
            self.esm_model.eval()
            self.esm_model.to(self.device)
            
            # Freeze the model
            for param in self.esm_model.parameters():
                param.requires_grad = False
            
            print(f"Successfully loaded {self.model_variant} (frozen)")
            
            # Test the model
            test_sequence = ("test_seq", "MKFLVNVALVFMVVYISYIY")
            batch_converter = self.tokenizer.get_batch_converter()
            batch_labels, batch_strings, batch_tokens = batch_converter([test_sequence])
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                test_output = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
                print(f"Model test successful - output shape: {test_output['representations'][6].shape}")
            
        except Exception as e:
            print(f"Error loading model {self.model_variant}: {e}")
            raise e
    
    def compute_engineered_features(self, sequence: str) -> np.ndarray:
        """
        Compute engineered residue features for charge prediction.
        
        Features:
        - nK: count of Lysine
        - nR: count of Arginine
        - nH: count of Histidine (weighted)
        - nD: count of Aspartic acid
        - nE: count of Glutamic acid
        - length: peptide length
        - net_basicity: nK + nR + w*nH - (nD + nE)
        
        Args:
            sequence: Peptide sequence
            
        Returns:
            Feature vector [nK, nR, nH, nD, nE, length, net_basicity]
        """
        seq_upper = sequence.upper()
        
        # Count residues
        nK = seq_upper.count('K')
        nR = seq_upper.count('R')
        nH = seq_upper.count('H')
        nD = seq_upper.count('D')
        nE = seq_upper.count('E')
        
        # Peptide length
        length = len(sequence)
        
        # Net basicity (weighted histidine with weight 0.5)
        w = 0.5  # Histidine weight
        net_basicity = nK + nR + w * nH - (nD + nE)
        
        return np.array([nK, nR, nH, nD, nE, length, net_basicity], dtype=np.float32)
    
    def read_sequence_data(self, file_path: str, 
                          sequence_column: int = 1,
                          delimiter: str = '\t', 
                          skip_header: bool = True) -> List[List[str]]:
        """
        Read peptide sequences from a CSV/TSV file.
        
        Args:
            file_path: Path to the input file
            sequence_column: Column index containing sequences (0-based indexing)
            delimiter: File delimiter ('\t' for TSV, ',' for CSV)
            skip_header: Whether to skip the first row (header row)
            
        Returns:
            List of data rows
        """
        print(f"Reading sequence data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            raw_data = [row for row in csv_reader]
        
        if skip_header and raw_data:
            header = raw_data[0]
            print(f"Header row: {header}")
            print(f"Sequence column index: {sequence_column}")
            if sequence_column < len(header):
                print(f"Sequence column name: '{header[sequence_column]}'")
            raw_data = raw_data[1:]
        
        print(f"Loaded {len(raw_data)} sequences")
        return raw_data
    
    def process_batch(self, batch_data: List[Tuple[str, str]]) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """
        Process a batch of sequences and extract features.
        
        Args:
            batch_data: List of (sequence_id, sequence) tuples
            
        Returns:
            Tuple of (esm_embeddings, engineered_features)
        """
        try:
            # Validate input data
            if not batch_data:
                raise ValueError("Empty batch data provided")
            
            # Prepare sequences
            sequences = []
            for seq_id, sequence in batch_data:
                if not sequence or not sequence.strip():
                    print(f"Warning: Empty sequence found for ID: {seq_id}")
                    continue
                
                # Validate sequence
                if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence.upper()):
                    print(f"Warning: Invalid amino acid characters in sequence {seq_id}: {sequence}")
                    continue
                
                sequences.append((seq_id, sequence.strip().upper()))
                self.extraction_stats["sequence_lengths"].append(len(sequence.strip()))
            
            if not sequences:
                raise ValueError("No valid sequences found in batch")
            
            # Convert sequences to model format
            batch_converter = self.tokenizer.get_batch_converter()
            batch_labels, batch_strings, batch_tokens = batch_converter(sequences)
            
            # Move to device
            batch_tokens = batch_tokens.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                # Determine layer number based on model variant
                if "esm2" in self.model_variant:
                    if "t6" in self.model_variant:
                        layer_num = 6
                    elif "t12" in self.model_variant:
                        layer_num = 12
                    elif "t30" in self.model_variant:
                        layer_num = 30
                    elif "t33" in self.model_variant:
                        layer_num = 33
                    elif "t36" in self.model_variant:
                        layer_num = 36
                    elif "t48" in self.model_variant:
                        layer_num = 48
                    else:
                        layer_num = 6
                else:
                    layer_num = 33
                
                model_output = self.esm_model(batch_tokens, repr_layers=[layer_num], return_contacts=False)
                
                # Use the requested layer
                if layer_num in model_output['representations']:
                    token_embeddings = model_output["representations"][layer_num]
                else:
                    last_layer = max(model_output['representations'].keys())
                    token_embeddings = model_output["representations"][last_layer]
            
            # Get sequence lengths (excluding special tokens)
            sequence_lengths = [len(seq) for _, seq in sequences]
            
            # Apply aggregation strategy
            aggregated_features = self._aggregate_sequence_features(token_embeddings, batch_tokens, sequence_lengths)
            
            # Compute engineered features
            engineered_features = []
            for _, sequence in sequences:
                eng_feat = self.compute_engineered_features(sequence)
                engineered_features.append(eng_feat)
            
            return aggregated_features, engineered_features
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            raise e
    
    def _aggregate_sequence_features(self, token_embeddings: torch.Tensor, 
                                   batch_tokens: torch.Tensor,
                                   sequence_lengths: List[int]) -> List[torch.Tensor]:
        """
        Apply the specified aggregation strategy to sequence embeddings.
        Uses attention mask-aware mean pooling.
        """
        if self.aggregation_method == "global_mean":
            return self._global_mean_pooling(token_embeddings, batch_tokens, sequence_lengths)
        elif self.aggregation_method == "global_max":
            return self._global_max_pooling(token_embeddings, batch_tokens, sequence_lengths)
        elif self.aggregation_method == "attention_weighted":
            return self._attention_weighted_pooling(token_embeddings, batch_tokens, sequence_lengths)
        else:
            # Default to global mean pooling
            return self._global_mean_pooling(token_embeddings, batch_tokens, sequence_lengths)
    
    def _global_mean_pooling(self, token_embeddings: torch.Tensor, 
                           batch_tokens: torch.Tensor,
                           sequence_lengths: List[int]) -> List[torch.Tensor]:
        """Apply attention mask-aware mean pooling to sequence embeddings."""
        pooled_features = []
        batch_size = token_embeddings.size(0)
        
        for i, seq_len in enumerate(sequence_lengths):
            # Create attention mask (1 for real tokens, 0 for padding)
            # ESM uses <cls> at position 0, <eos> at position seq_len+1
            # We want positions 1 to seq_len (actual sequence tokens)
            attention_mask = (batch_tokens[i] != self.tokenizer.padding_idx).float()
            
            # Extract sequence tokens (skip <cls> at position 0)
            # Sequence tokens are at positions 1 to seq_len
            seq_tokens = token_embeddings[i, 1:seq_len+1]  # [seq_len, embedding_dim]
            seq_mask = attention_mask[1:seq_len+1]  # [seq_len]
            
            # Mean pooling with attention mask
            if seq_mask.sum() > 0:
                masked_embeddings = seq_tokens * seq_mask.unsqueeze(1)
                pooled_feature = masked_embeddings.sum(dim=0) / seq_mask.sum()
            else:
                # Fallback to simple mean if mask is empty
                pooled_feature = seq_tokens.mean(dim=0)
            
            pooled_features.append(pooled_feature)
        
        return pooled_features
    
    def _global_max_pooling(self, token_embeddings: torch.Tensor, 
                          batch_tokens: torch.Tensor,
                          sequence_lengths: List[int]) -> List[torch.Tensor]:
        """Apply max pooling to sequence embeddings."""
        pooled_features = []
        
        for i, seq_len in enumerate(sequence_lengths):
            seq_tokens = token_embeddings[i, 1:seq_len+1]
            pooled_feature = seq_tokens.max(dim=0)[0]
            pooled_features.append(pooled_feature)
        
        return pooled_features
    
    def _attention_weighted_pooling(self, token_embeddings: torch.Tensor, 
                                  batch_tokens: torch.Tensor,
                                  sequence_lengths: List[int]) -> List[torch.Tensor]:
        """Apply attention-weighted pooling to sequence embeddings."""
        pooled_features = []
        
        for i, seq_len in enumerate(sequence_lengths):
            seq_tokens = token_embeddings[i, 1:seq_len+1]
            
            # Simple attention mechanism
            attention_scores = torch.softmax(
                torch.sum(seq_tokens * seq_tokens, dim=1), dim=0
            )
            weighted_features = seq_tokens * attention_scores.unsqueeze(1)
            pooled_feature = weighted_features.sum(dim=0)
            pooled_features.append(pooled_feature)
        
        return pooled_features
    
    def extract_features_from_file(self, input_file_path: str, 
                                  output_file_path: str,
                                  sequence_column: int = 1,
                                  delimiter: str = '\t',
                                  skip_header: bool = True) -> None:
        """
        Extract features from all sequences in a file.
        
        Args:
            input_file_path: Path to input CSV/TSV file
            output_file_path: Path to save extracted features
            sequence_column: Column index containing sequences (0-based indexing)
            delimiter: File delimiter ('\t' for TSV, ',' for CSV)
            skip_header: Whether to skip the first row (header row)
        """
        # Load model
        self.load_esm_model()
        
        # Read data
        raw_sequence_data = self.read_sequence_data(
            input_file_path, sequence_column, delimiter, skip_header
        )
        
        # Calculate number of batches
        total_sequences = len(raw_sequence_data)
        num_batches = (total_sequences // self.processing_batch_size) + 1
        
        print(f"Processing {total_sequences} sequences in {num_batches} batches")
        print(f"Using aggregation strategy: {self.aggregation_method}")
        
        all_esm_features = []
        all_engineered_features = []
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.processing_batch_size
            end_idx = min((batch_idx + 1) * self.processing_batch_size, total_sequences)
            
            if start_idx >= total_sequences:
                break
            
            batch_data = raw_sequence_data[start_idx:end_idx]
            
            # Prepare batch for processing
            batch_sequences = []
            for i, row in enumerate(batch_data):
                sequence_id = f"peptide_{start_idx + i}"
                sequence_text = row[sequence_column]
                batch_sequences.append((sequence_id, sequence_text))
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} "
                  f"(sequences {start_idx + 1}-{end_idx})")
            
            # Extract features for this batch
            batch_esm_features, batch_engineered_features = self.process_batch(batch_sequences)
            all_esm_features.extend(batch_esm_features)
            all_engineered_features.extend(batch_engineered_features)
        
        # Normalize engineered features across the dataset
        if all_engineered_features:
            engineered_array = np.array(all_engineered_features)
            # Normalize each feature (zero mean, unit variance)
            mean = engineered_array.mean(axis=0)
            std = engineered_array.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            normalized_engineered = (engineered_array - mean) / std
            
            # Save normalization parameters
            normalization_params = {
                'mean': mean.tolist(),
                'std': std.tolist()
            }
            
            # Convert back to list of arrays
            all_engineered_features = [normalized_engineered[i] for i in range(len(normalized_engineered))]
        else:
            normalization_params = None
        
        # Save results
        results = {
            'esm_features': all_esm_features,
            'engineered_features': all_engineered_features,
            'normalization_params': normalization_params
        }
        
        torch.save(results, output_file_path)
        print(f"Saved feature vectors to: {output_file_path}")
        
        # Print statistics
        self._print_extraction_statistics()
        
        print(f"Feature extraction completed successfully!")
        print(f"Total sequences processed: {len(all_esm_features)}")
        if all_esm_features:
            print(f"ESM feature dimension: {all_esm_features[0].size()}")
            print(f"Engineered feature dimension: {len(all_engineered_features[0])}")
    
    def _print_extraction_statistics(self) -> None:
        """Print statistics about the extraction process."""
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION STATISTICS")
        print("=" * 60)
        
        print(f"Total sequences processed: {len(self.extraction_stats['sequence_lengths'])}")
        
        if self.extraction_stats['sequence_lengths']:
            lengths = self.extraction_stats['sequence_lengths']
            print(f"Sequence length - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.2f}")
        
        print("=" * 60)


def main():
    """
    Main function demonstrating usage of the ESMFeatureExtractor.
    """
    # Configuration parameters 
    input_data_path = '/hpc/shared/uu_immunopeptidomics/CCS_project_part1/revised_ccs_codes/ccs_mz_correlation_data2.tsv'
    output_features_path = '/hpc/shared/uu_immunopeptidomics/features_hla1.pt'
    
    # ESM-2 model selection
    selected_model = "esm2_t6_8M_UR50D"
    
    # Aggregation strategy selection
    selected_strategy = "global_mean"
    
    # Processing parameters
    batch_processing_size = 20000
    
    print("=" * 80)
    print("ESM-2 FEATURE EXTRACTION WITH ENGINEERED FEATURES")
    print("=" * 80)
    print(f"Model: {selected_model}")
    print(f"Aggregation Strategy: {selected_strategy}")
    print(f"Batch Size: {batch_processing_size}")
    print("=" * 80)
    
    # Initialize feature extractor
    feature_extractor = ESMFeatureExtractor(
        model_type=selected_model,
        aggregation_strategy=selected_strategy,
        batch_size=batch_processing_size
    )
    
    # Extract features
    feature_extractor.extract_features_from_file(
        input_file_path=input_data_path,
        output_file_path=output_features_path,
        sequence_column=1,  # Column containing peptide sequences
        delimiter='\t',
        skip_header=True
    )


if __name__ == "__main__":
    main()
