"""
@author: a-nakai-k (modified by Claude)

Code for preprocessing of peptide sequences with ESM-1b.
Prepare csv file with column of sequence data.
This code outputs temporal sequences named 'sequenceTensor1/2/... .pt' to save memory.
The resulting processed sequence (all sequences) is saved in 'sequenceTensor_a**b**gamma**.pt'.
Please remove the temporal files if necessary.
"""

import torch
import csv
import numpy as np

# parameters
data_path = '/content/CCS_project_part1/processed_data/immunopeptide_data_JPST002044.tsv'
column_idx = 0              # Changed from 1 to 0 - the peptide sequence should be in the first column
a = 1000                    # parameter for positional encoding
b = 1                       # parameter for positional encoding
gamma = 0                   # parameter for positional encoding

# Load and validate data first
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acid letters

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
        seq = row[column_idx].strip()
        if not seq or not all(aa in valid_amino_acids for aa in seq.upper()):
            print(f"Skipping row {i+1}, invalid sequence: '{seq}'")
            skipped += 1
            continue
            
        data.append(row)
    
    print(f"Loaded {len(data)} valid sequences, skipped {skipped} invalid entries")

# Safeguard against empty dataset
if not data:
    raise ValueError("No valid peptide sequences found in the input file!")

print("File validated and opened")

# load ESM-1b model
datasize = 20000            # batch size for preprocessing to save memory
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
batch_converter = alphabet.get_batch_converter()
nIteration = len(data)//datasize + 1

print("Pretrained model loaded")

# apply positional encoding and save processed sequences
for itr in range(nIteration):
    print(f"Processing batch {itr+1}/{nIteration}")
    if itr == nIteration-1:
        datalist = data[itr*datasize:]
    else:
        datalist = data[itr*datasize:(itr+1)*datasize]
    
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
        continue
        
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)
    except Exception as e:
        print(f"Error during batch conversion: {e}")
        print(f"First 5 samples of problematic batch: {seqdata[:5]}")
        continue

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations with positional encoding
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []   # list of tensors
    pos_enc = torch.zeros(token_representations.size()[1], token_representations.size()[2])
    for pos in range(pos_enc.size()[0]):
        for i in range(pos_enc.size()[1]):
            if i % 2 == 0:
                pos_enc[pos, i] = (np.sin((pos+1)/(a**(i/pos_enc.size()[1]))))**b + gamma
            else:
                pos_enc[pos, i] = (np.cos((pos+1)/(a**((i-1)/pos_enc.size()[1]))))**b + gamma
                
    for i, (_, seq) in enumerate(seqdata):
        seq_len = len(seq)
        if seq_len < 2:  # Need at least 2 characters for N and C terminal processing
            print(f"Sequence too short (length={seq_len}): {seq}")
            continue
            
        mid_point = round(seq_len/2)
        tmp_repre_n = token_representations[i, 1:mid_point+1]
        tmp_repre_c = token_representations[i, mid_point+1:seq_len+1]
        
        # Prevent dimension errors with very short sequences
        if tmp_repre_n.size(0) == 0 or tmp_repre_c.size(0) == 0:
            print(f"Skipping sequence with problematic split: {seq}")
            continue
            
        # Ensure positional encoding matches the sequence portion lengths
        n_pos_enc = pos_enc[:tmp_repre_n.size(0)]
        c_pos_enc = torch.flip(pos_enc[:tmp_repre_c.size(0)], [0])  # Reversed for C-terminal
        
        try:
            nseq = torch.mul(tmp_repre_n, n_pos_enc).mean(0)
            cseq = torch.mul(tmp_repre_c, c_pos_enc).mean(0)
            sequence_representations.append(torch.cat((nseq, cseq)))
        except Exception as e:
            print(f"Error processing sequence {i}: {seq}, error: {e}")
            print(f"N-terminal shape: {tmp_repre_n.shape}, N-pos shape: {n_pos_enc.shape}")
            print(f"C-terminal shape: {tmp_repre_c.shape}, C-pos shape: {c_pos_enc.shape}")
    
    # Save batch results
    torch.save(sequence_representations, f'sequenceTensor{itr+1}.pt')
    print(f"Saved batch {itr+1} with {len(sequence_representations)} processed sequences")

# create resulting .pt file
print("Combining all batches...")
sequence_representations = []
for itr in range(nIteration):
    try:
        batch_file = f'sequenceTensor{itr+1}.pt'
        if os.path.exists(batch_file):
            batch_data = torch.load(batch_file)
            sequence_representations.extend(batch_data)
            print(f"Added {len(batch_data)} sequences from batch {itr+1}")
    except Exception as e:
        print(f"Error loading batch {itr+1}: {e}")

output_file = f'sequenceTensor_a{a}b{b}gamma{gamma}.pt'
torch.save(sequence_representations, output_file)
print(f"Successfully saved {len(sequence_representations)} sequences to {output_file}")
