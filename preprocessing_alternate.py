"""
@author: a-nakai-k

Code for preprocessing of peptide sequences with ESM-1b.
Prepare csv file with column of sequence data.
This code outputs temporal sequences named 'sequenceTensor1/2/... .pt' to save memory.
The resulting processed sequence (all sequences) is saved in 'sequenceTensor_a**b**gamma**.pt'.
Please remove the temporal files if necessary.

Modified to handle peptides with modifications by filtering to standard amino acids only.
"""

import torch
import csv
import numpy as np
import re

# parameters
#data_path = './data.csv'    # path to csv file
data_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1_new.tsv'
column_idx = 1            # column index of sequence data in csv file
a = 1000                    # parameter for positional encoding
b = 1                       # parameter for positional encoding
gamma = 0                   # parameter for positional encoding

def clean_peptide_sequence(sequence):
    """
    Clean peptide sequence to contain only standard amino acids.
    Removes modifications, non-amino acid characters, and converts to uppercase.
    
    Args:
        sequence (str): Raw peptide sequence
        
    Returns:
        str: Cleaned sequence with only standard amino acids
    """
    if not sequence or not isinstance(sequence, str):
        return ""
    
    # Standard 20 amino acids
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Remove common modification patterns
    # Remove parenthetical modifications like (+15.99), (-17.03), etc.
    cleaned = re.sub(r'\([^)]*\)', '', sequence)
    
    # Remove square bracket modifications like [+15], [-17], etc.
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # Remove numbers and special characters commonly used in modifications
    cleaned = re.sub(r'[0-9+\-.,;:_\s]', '', cleaned)
    
    # Convert to uppercase
    cleaned = cleaned.upper()
    
    # Keep only standard amino acids
    cleaned = ''.join([char for char in cleaned if char in standard_aa])
    
    return cleaned

# file open
with open(data_path) as f:
    reader = csv.reader(f, delimiter='\t')
    data = [row for row in reader]
del(data[0])    # remove label, if necessary

print("File opened")

# Clean and validate sequences
cleaned_data = []
skipped_sequences = 0

for i, row in enumerate(data):
    if len(row) > column_idx:
        original_seq = row[column_idx]
        cleaned_seq = clean_peptide_sequence(original_seq)
        
        # Skip sequences that are too short after cleaning (less than 3 amino acids)
        if len(cleaned_seq) >= 3:
            row[column_idx] = cleaned_seq
            cleaned_data.append(row)
        else:
            skipped_sequences += 1
            print(f"Skipped sequence {i+1}: '{original_seq}' -> '{cleaned_seq}' (too short after cleaning)")
    else:
        skipped_sequences += 1
        print(f"Skipped row {i+1}: insufficient columns")

data = cleaned_data
print(f"Sequences processed: {len(data)}")
print(f"Sequences skipped: {skipped_sequences}")

# Show some examples of cleaned sequences
if len(data) > 0:
    print("\nFirst few cleaned sequences:")
    for i in range(min(5, len(data))):
        print(f"  {i+1}: {data[i][column_idx]}")

# load ESM-1b model
datasize = 20000            # batch size for preprocessing to save memory
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")   # not need to clone and/or install ESM
batch_converter = alphabet.get_batch_converter()
nIteration = len(data)//datasize + 1

print("Pretrained model loaded")

# apply positional encoding and save processed sequences
for itr in range(nIteration):
    # print(itr)
    if itr == nIteration-1:
        datalist = data[itr*datasize:]
    else:
        datalist = data[itr*datasize:(itr+1)*datasize]
    
    if len(datalist) == 0:  # Skip empty batches
        continue
        
    seqdata = []
    for i in range(len(datalist)):
        sequence = datalist[i][column_idx]
        # Double-check sequence is clean before processing
        if sequence and len(sequence) > 0:
            seqdata.append(("protein"+str(i), sequence))
    
    if len(seqdata) == 0:  # Skip if no valid sequences in this batch
        continue

    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)
    except Exception as e:
        print(f"Error in batch {itr+1}: {e}")
        print("Problematic sequences in this batch:")
        for j, (label, seq) in enumerate(seqdata):
            print(f"  {label}: '{seq}'")
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
        tmp_repre_n = token_representations[i, 1 : round(len(seq)/2) + 1]
        tmp_repre_c = token_representations[i, round(len(seq)/2) + 1 : len(seq) + 1]
        nseq = torch.mul(tmp_repre_n, pos_enc[0:round(len(seq)/2), :]).mean(0)
        cseq = torch.mul(tmp_repre_c, torch.flip(pos_enc, [0])[-len(seq)+round(len(seq)/2):, :]).mean(0)
        sequence_representations.append(torch.cat((nseq, cseq)))
    
    torch.save(sequence_representations, 'sequenceTensor'+str(itr+1)+'.pt')
    print(f"Processed batch {itr+1}/{nIteration}")

# create resulting .pt file
sequence_representations = []
for itr in range(nIteration):
    try:
        batch_data = torch.load('sequenceTensor'+str(itr+1)+'.pt')
        sequence_representations.extend(batch_data)
    except FileNotFoundError:
        print(f"Warning: sequenceTensor{itr+1}.pt not found, skipping...")
        continue

if len(sequence_representations) > 0:
    torch.save(sequence_representations, '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_testnewdata_a'+str(a)+'b'+str(b)+'gamma'+str(gamma)+'.pt')
    print(f"Final tensor saved with {len(sequence_representations)} sequences")
else:
    print("No sequences were successfully processed!")
