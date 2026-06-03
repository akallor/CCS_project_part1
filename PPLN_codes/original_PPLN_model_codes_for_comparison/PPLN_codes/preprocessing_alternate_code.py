"""
Lightweight Peptide Feature Extraction

This script extracts features from peptide sequences using either:
1. Simple amino acid physicochemical properties (very fast)
2. A lightweight protein language model (if available)

The script is designed to work efficiently even on low-resource environments.
"""

import torch
import csv
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import pickle
from collections import Counter

# Define amino acid properties
# Format: [hydrophobicity, size, charge, polarity, aromaticity]
AA_PROPS = {
    'A': [1.8, 0.31, 0, 0, 0],  # Alanine
    'C': [2.5, 0.36, 0, 0, 0],  # Cysteine
    'D': [-3.5, 0.38, -1, 1, 0],  # Aspartic acid
    'E': [-3.5, 0.42, -1, 1, 0],  # Glutamic acid
    'F': [2.8, 0.58, 0, 0, 1],  # Phenylalanine
    'G': [-0.4, 0.25, 0, 0, 0],  # Glycine
    'H': [-3.2, 0.46, 0.5, 1, 1],  # Histidine
    'I': [4.5, 0.47, 0, 0, 0],  # Isoleucine
    'K': [-3.9, 0.49, 1, 1, 0],  # Lysine
    'L': [3.8, 0.47, 0, 0, 0],  # Leucine
    'M': [1.9, 0.50, 0, 0, 0],  # Methionine
    'N': [-3.5, 0.42, 0, 1, 0],  # Asparagine
    'P': [-1.6, 0.36, 0, 0, 0],  # Proline
    'Q': [-3.5, 0.42, 0, 1, 0],  # Glutamine
    'R': [-4.5, 0.59, 1, 1, 0],  # Arginine
    'S': [-0.8, 0.33, 0, 1, 0],  # Serine
    'T': [-0.7, 0.39, 0, 1, 0],  # Threonine
    'V': [4.2, 0.39, 0, 0, 0],  # Valine
    'W': [-0.9, 0.65, 0, 0, 1],  # Tryptophan
    'Y': [-1.3, 0.64, 0, 1, 1],  # Tyrosine
}

# Command line arguments
parser = argparse.ArgumentParser(description='Extract features from peptide sequences.')
parser.add_argument('--data_path', type=str, 
                    default='/content/CCS_project_part1/processed_data/immunopeptide_data_JPST002044.tsv',
                    help='Path to input data file')
parser.add_argument('--column_idx', type=int, default=0, 
                    help='Column index of peptide sequences')
parser.add_argument('--output_file', type=str, 
                    default='peptide_features.pt',
                    help='Output file for extracted features')
parser.add_argument('--batch_size', type=int, default=1000, 
                    help='Batch size for processing')
parser.add_argument('--use_esm', action='store_true',
                    help='Try to use a lightweight ESM model (ESM-2) if available')
parser.add_argument('--a', type=float, default=1000, help='Parameter a for positional encoding')
parser.add_argument('--b', type=float, default=1, help='Parameter b for positional encoding')
parser.add_argument('--gamma', type=float, default=0, help='Parameter gamma for positional encoding')
                    
args = parser.parse_args()

# Parameters
data_path = args.data_path
column_idx = args.column_idx
output_file = args.output_file
batch_size = args.batch_size
use_esm = args.use_esm
a = args.a
b = args.b
gamma = args.gamma

print(f"Processing parameters: output_file={output_file}, use_esm={use_esm}")

# Function to check if file exists
def is_valid_file(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

# Load data
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acid letters

# Load or create cached processed data
data_cache_file = f"{data_path}.processed_cache.pkl"
if is_valid_file(data_cache_file):
    print(f"Loading preprocessed data from cache {data_cache_file}...")
    with open(data_cache_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} valid sequences from cache")
else:
    print(f"Loading and preprocessing data from {data_path}...")
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Store the header row
        print(f"Header: {header}")
        
        # Filter valid sequences
        data = []
        skipped = 0
        for i, row in enumerate(reader):
            if len(row) <= column_idx:
                print(f"Row {i+1} doesn't have enough columns: {row}")
                skipped += 1
                continue
                
            # Ensure we're working with a valid peptide sequence
            seq = row[column_idx].strip().upper()
            if not seq or not all(aa in valid_amino_acids for aa in seq):
                print(f"Skipping row {i+1}, invalid sequence: '{seq}'")
                skipped += 1
                continue
                
            data.append((seq, row))  # Store sequence and original row
        
        print(f"Loaded {len(data)} valid sequences, skipped {skipped} invalid entries")
        
        # Cache preprocessed data
        with open(data_cache_file, 'wb') as f:
            pickle.dump(data, f)

# Analyze sequence properties
seq_lengths = [len(seq) for seq, _ in data]
print(f"Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.2f}")
aa_counter = Counter()
for seq, _ in data:
    aa_counter.update(seq)
print("Amino acid distribution:")
for aa, count in aa_counter.most_common():
    print(f"  {aa}: {count} ({count/sum(aa_counter.values())*100:.2f}%)")

# Try to load ESM-2 if requested
esm_model = None
if use_esm:
    try:
        print("Trying to load lightweight ESM-2 (3B) model...")
        import esm
        # Load the smallest ESM-2 model (35M parameters) - much smaller than ESM-1b
        esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        esm_model.eval()  # Set to evaluation mode
        batch_converter = alphabet.get_batch_converter()
        print("ESM-2 model loaded successfully")
    except Exception as e:
        print(f"Failed to load ESM-2 model: {e}")
        print("Falling back to physicochemical features only")
        esm_model = None

# Function to extract features using ESM-2
def extract_esm_features(sequences, batch_size=32):
    features = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM features", total=total_batches):
        batch_seqs = sequences[i:i+batch_size]
        batch_labels = [f"protein_{j}" for j in range(len(batch_seqs))]
        batch_pairs = list(zip(batch_labels, batch_seqs))
        
        try:
            with torch.no_grad():
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_pairs)
                results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
                token_representations = results["representations"][6]
                
                # Average the token representations (excluding special tokens)
                for j, seq in enumerate(batch_seqs):
                    seq_len = len(seq)
                    seq_rep = token_representations[j, 1:seq_len+1].mean(0)
                    features.append(seq_rep)
        except Exception as e:
            print(f"Error in ESM batch processing: {e}")
            # Fall back to physicochemical properties
            for seq in batch_seqs:
                features.append(extract_physicochemical_features(seq))
    
    return features

# Function to extract physicochemical features
def extract_physicochemical_features(sequence):
    # Number of features per position
    n_props = len(list(AA_PROPS.values())[0])
    
    # Initialize feature vector
    # We'll use various statistics to capture sequence properties
    feat_vec = np.zeros(n_props * 8)  # mean, std, min, max, N-term, C-term, mid, weighted
    
    # Extract position-specific properties
    pos_props = np.array([AA_PROPS[aa] for aa in sequence])
    
    # Basic statistics
    feat_vec[0:n_props] = np.mean(pos_props, axis=0)
    feat_vec[n_props:2*n_props] = np.std(pos_props, axis=0)
    feat_vec[2*n_props:3*n_props] = np.min(pos_props, axis=0)
    feat_vec[3*n_props:4*n_props] = np.max(pos_props, axis=0)
    
    # Terminal properties (more weighted in biological activity)
    n_term_size = min(3, len(sequence))
    c_term_size = min(3, len(sequence))
    mid_point = len(sequence) // 2
    mid_size = min(3, len(sequence))
    
    if n_term_size > 0:
        feat_vec[4*n_props:5*n_props] = np.mean(pos_props[:n_term_size], axis=0)
    if c_term_size > 0:
        feat_vec[5*n_props:6*n_props] = np.mean(pos_props[-c_term_size:], axis=0)
    if mid_size > 0:
        mid_start = max(0, mid_point - mid_size//2)
        mid_end = min(len(sequence), mid_point + mid_size//2 + 1)
        feat_vec[6*n_props:7*n_props] = np.mean(pos_props[mid_start:mid_end], axis=0)
    
    # Position-weighted features (similar to positional encoding concept)
    weights = np.zeros(len(sequence))
    for i in range(len(sequence)):
        pos = i + 1
        # Similar to the positional encoding in the original code
        weight_val = (np.sin(pos/(a**(pos/len(sequence)))))**b + gamma
        weights[i] = weight_val
    
    # Normalize weights
    if len(weights) > 0:
        weights = weights / np.sum(weights)
        weighted_props = np.sum(pos_props * weights[:, np.newaxis], axis=0)
        feat_vec[7*n_props:8*n_props] = weighted_props
    
    # Additional derived features for peptides
    # Add simple sequence-based features
    extra_feats = np.zeros(10)
    extra_feats[0] = len(sequence)  # length
    
    # Amino acid class counts
    hydrophobic = sum(1 for aa in sequence if aa in "AVILMFYWC")
    polar = sum(1 for aa in sequence if aa in "STNQY")
    charged_pos = sum(1 for aa in sequence if aa in "RHK")
    charged_neg = sum(1 for aa in sequence if aa in "DE")
    special = sum(1 for aa in sequence if aa in "PG")
    
    # Normalized counts (percentage of sequence)
    if len(sequence) > 0:
        extra_feats[1] = hydrophobic / len(sequence)
        extra_feats[2] = polar / len(sequence)
        extra_feats[3] = charged_pos / len(sequence)
        extra_feats[4] = charged_neg / len(sequence)
        extra_feats[5] = special / len(sequence)
    
    # Net charge
    extra_feats[6] = charged_pos - charged_neg
    
    # Peptide density approximation
    avg_mass = sum({'A': 71.08, 'C': 103.14, 'D': 115.09, 'E': 129.12, 'F': 147.18,
                    'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.17, 'L': 113.16,
                    'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19,
                    'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.21, 'Y': 163.18}.get(aa, 0) 
                  for aa in sequence) / len(sequence) if len(sequence) > 0 else 0
    extra_feats[7] = avg_mass
    
    # Hydrophobicity ratio (N-terminal vs C-terminal)
    n_term = sequence[:len(sequence)//2]
    c_term = sequence[len(sequence)//2:]
    n_hydrophobic = sum(1 for aa in n_term if aa in "AVILMFYWC") / len(n_term) if len(n_term) > 0 else 0
    c_hydrophobic = sum(1 for aa in c_term if aa in "AVILMFYWC") / len(c_term) if len(c_term) > 0 else 0
    extra_feats[8] = n_hydrophobic / c_hydrophobic if c_hydrophobic > 0 else 1.0
    
    # Charge difference (N-terminal vs C-terminal)
    n_charge = sum(1 for aa in n_term if aa in "RHK") - sum(1 for aa in n_term if aa in "DE")
    c_charge = sum(1 for aa in c_term if aa in "RHK") - sum(1 for aa in c_term if aa in "DE")
    extra_feats[9] = n_charge - c_charge
    
    # Combine all features
    return np.concatenate([feat_vec, extra_feats])

# Process sequences and extract features
print(f"Extracting features from {len(data)} sequences...")
if esm_model and use_esm:
    # Extract sequences only
    sequences = [seq for seq, _ in data]
    features = extract_esm_features(sequences, batch_size=batch_size)
    # Convert PyTorch tensors to numpy arrays
    features = [f.numpy() if isinstance(f, torch.Tensor) else f for f in features]
else:
    # Use physicochemical features
    sequences = [seq for seq, _ in data]
    features = []
    for idx, seq in enumerate(tqdm(sequences, desc="Extracting physicochemical features")):
        features.append(extract_physicochemical_features(seq))

# Associate features with original data rows
print(f"Generated {len(features)} feature vectors, shape: {features[0].shape}")
feature_data = [(features[i], data[i][1]) for i in range(len(features))]

# Save results according to specified output format
print(f"Saving results to {output_file}...")
if output_file.endswith('.pt'):
    torch.save(feature_data, output_file)
elif output_file.endswith('.npy'):
    np.save(output_file, feature_data)
else:
    # Default to pickle format
    with open(output_file, 'wb') as f:
        pickle.dump(feature_data, f)

print(f"Successfully saved {len(feature_data)} processed sequences to {output_file}")
print("Processing complete!")

# Optional: Create visualization of feature distributions
try:
    import matplotlib.pyplot as plt
    
    # Convert features to numpy array for easier handling
    feat_array = np.array(features)
    
    # Plot feature distributions
    plt.figure(figsize=(12, 8))
    
    # Plot mean of each feature
    means = np.mean(feat_array, axis=0)
    plt.bar(range(len(means)), means)
    plt.title('Feature Means')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.savefig('feature_distribution.png')
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = np.corrcoef(feat_array, rowvar=False)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.savefig('feature_correlation.png')
    
    print("Generated visualization plots: feature_distribution.png, feature_correlation.png")
except Exception as e:
    print(f"Could not generate visualizations: {e}")
