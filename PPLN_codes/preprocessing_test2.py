"""
@author: a-nakai-k

Code for preprocessing of peptide sequences with ESM-1b.
Prepare csv file with column of sequence data.
This code outputs temporal sequences named 'sequenceTensor1/2/... .pt' to save memory.
The resulting processed sequence (all sequences) is saved in 'sequenceTensor_a**b**gamma**.pt'.
Please remove the temporal files if necessary.
"""

import torch
import csv
import numpy as np
import os
import time
import argparse
from tqdm import tqdm

# Parameters via command line arguments
parser = argparse.ArgumentParser(description='Process peptide sequences with ESM-1b.')
parser.add_argument('--data_path', type=str, default='/content/CCS_project_part1/processed_data/immunopeptide_data_JPST002044.tsv',
                    help='Path to input data file')
parser.add_argument('--column_idx', type=int, default=0, 
                    help='Column index of peptide sequences')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='Batch size for ESM model processing (reduce this if memory issues)')
parser.add_argument('--start_batch', type=int, default=0,
                    help='Batch number to start from (for resuming interrupted runs)')
parser.add_argument('--a', type=float, default=1000,
                    help='Parameter a for positional encoding')
parser.add_argument('--b', type=float, default=1,
                    help='Parameter b for positional encoding')
parser.add_argument('--gamma', type=float, default=0,
                    help='Parameter gamma for positional encoding')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run on (cpu, cuda, cuda:0, etc.)')

args = parser.parse_args()

# Parameters from command line or defaults
data_path = args.data_path
column_idx = args.column_idx
a = args.a
b = args.b
gamma = args.gamma
device = args.device
batch_size = args.batch_size
start_batch = args.start_batch

print(f"Processing parameters: a={a}, b={b}, gamma={gamma}, device={device}, batch_size={batch_size}")

# Function to check if file exists and is non-empty
def is_valid_file(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

# Check if the combined result file already exists
result_file = f'sequenceTensor_a{a}b{b}gamma{gamma}.pt'
if is_valid_file(result_file):
    print(f"Result file {result_file} already exists. Please rename or remove it to reprocess.")
    print("Exiting...")
    exit(0)

# Load and validate data first
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acid letters

# Load data or use cached version for faster restart
data_cache_file = f"{data_path}.processed_cache.pt"
if is_valid_file(data_cache_file) and start_batch > 0:
    print(f"Loading preprocessed data from cache {data_cache_file}...")
    data = torch.load(data_cache_file)
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
                
            data.append(row)
        
        print(f"Loaded {len(data)} valid sequences, skipped {skipped} invalid entries")
        
        # Cache preprocessed data for faster restart
        torch.save(data, data_cache_file)

# Safeguard against empty dataset
if not data:
    raise ValueError("No valid peptide sequences found in the input file!")

print("File validated and opened")

# Load ESM-1b model
print(f"Loading ESM-1b model to {device}...")
try:
    # Try to load model directly to the specified device
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    batch_converter = alphabet.get_batch_converter()
    print(f"Model loaded successfully to {device}")
except Exception as e:
    print(f"Error loading model to {device}: {e}")
    print("Falling back to CPU...")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
    model.eval()
    device = 'cpu'
    batch_converter = alphabet.get_batch_converter()

# Calculate number of batches based on actual batch size
nIteration = (len(data) + batch_size - 1) // batch_size

print(f"Pretrained model loaded on {device}")
print(f"Processing {len(data)} sequences in {nIteration} batches (batch size: {batch_size})")

# Generate positional encoding once for all batches
def get_positional_encoding(max_len, feature_dim, a, b, gamma):
    """Precompute positional encoding for efficiency"""
    pos_enc = torch.zeros(max_len, feature_dim)
    for pos in range(max_len):
        for i in range(feature_dim):
            if i % 2 == 0:
                pos_enc[pos, i] = (np.sin((pos+1)/(a**(i/feature_dim))))**b + gamma
            else:
                pos_enc[pos, i] = (np.cos((pos+1)/(a**((i-1)/feature_dim))))**b + gamma
    return pos_enc

# Find maximum sequence length to create appropriate positional encoding
max_seq_len = max(len(row[column_idx].strip()) for row in data) + 1  # +1 for safety
print(f"Maximum sequence length: {max_seq_len}")

# Create progress bar for batch processing
pbar = tqdm(total=nIteration-start_batch, desc="Processing batches")

# Check if we can resume from a specific batch
start_idx = start_batch * batch_size
processed_count = 0
total_sequences = 0

# apply positional encoding and save processed sequences
for itr in range(start_batch, nIteration):
    # Define batch checkpoint file
    batch_file = f'sequenceTensor{itr+1}.pt'
    
    # Skip if this batch is already processed
    if is_valid_file(batch_file) and start_batch > 0:
        print(f"Batch {itr+1} already processed, skipping...")
        pbar.update(1)
        continue
    
    # Extract batch data
    start_idx = itr * batch_size
    end_idx = min((itr + 1) * batch_size, len(data))
    datalist = data[start_idx:end_idx]
    
    # Prepare sequence data
    seqdata = []
    for i, row in enumerate(datalist):
        try:
            seq = row[column_idx].strip().upper()  # Standardize to uppercase
            if seq:  # Make sure we have a non-empty sequence
                seqdata.append(("protein"+str(i), seq))
        except Exception as e:
            print(f"Error processing row: {row}, error: {e}")
    
    # Skip empty batches
    if not seqdata:
        print(f"Batch {itr+1} is empty, skipping")
        pbar.update(1)
        continue
    
    # Process batch with model
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)
        batch_tokens = batch_tokens.to(device)
        
        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        # Move representations back to CPU for further processing
        if device != 'cpu':
            token_representations = token_representations.cpu()
        
        # Get the feature dimension
        feature_dim = token_representations.size()[2]
        
        # Generate positional encoding matrix
        max_batch_seq_len = token_representations.size()[1]
        pos_enc = get_positional_encoding(max_batch_seq_len, feature_dim, a, b, gamma)
        
        # Process each sequence in the batch
        sequence_representations = []
        for i, (_, seq) in enumerate(seqdata):
            seq_len = len(seq)
            if seq_len < 2:  # Need at least 2 characters for N and C terminal processing
                continue
                
            mid_point = round(seq_len/2)
            tmp_repre_n = token_representations[i, 1:mid_point+1]
            tmp_repre_c = token_representations[i, mid_point+1:seq_len+1]
            
            # Prevent dimension errors with very short sequences
            if tmp_repre_n.size(0) == 0 or tmp_repre_c.size(0) == 0:
                continue
                
            # Ensure positional encoding matches the sequence portion lengths
            n_pos_enc = pos_enc[:tmp_repre_n.size(0)]
            c_pos_enc = torch.flip(pos_enc[:tmp_repre_c.size(0)], [0])  # Reversed for C-terminal
            
            try:
                nseq = torch.mul(tmp_repre_n, n_pos_enc).mean(0)
                cseq = torch.mul(tmp_repre_c, c_pos_enc).mean(0)
                sequence_representations.append(torch.cat((nseq, cseq)))
                processed_count += 1
            except Exception as e:
                print(f"Error processing sequence: {seq}, error: {e}")
        
        # Save batch results
        torch.save(sequence_representations, batch_file)
        total_sequences += len(sequence_representations)
        
    except Exception as e:
        print(f"Error processing batch {itr+1}: {e}")
    
    # Update progress bar
    pbar.update(1)
    pbar.set_postfix({"processed": processed_count, "total": total_sequences})
    
    # Perform occasional garbage collection to free memory
    if itr % 5 == 0:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Close progress bar
pbar.close()

# Create resulting .pt file by combining all batches
print("\nCombining all processed batches...")
sequence_representations = []
combined_pbar = tqdm(total=nIteration)

for itr in range(nIteration):
    try:
        batch_file = f'sequenceTensor{itr+1}.pt'
        if is_valid_file(batch_file):
            batch_data = torch.load(batch_file)
            sequence_representations.extend(batch_data)
            combined_pbar.set_postfix({"sequences": len(sequence_representations)})
    except Exception as e:
        print(f"Error loading batch {itr+1}: {e}")
    combined_pbar.update(1)

combined_pbar.close()

# Save final result
output_file = f'sequenceTensor_a{a}b{b}gamma{gamma}.pt'
torch.save(sequence_representations, output_file)
print(f"Successfully saved {len(sequence_representations)} sequences to {output_file}")
print("Processing complete!")
