"""
@author: a-nakai-k (Enhanced to use GPUs)

Code for preprocessing of peptide sequences with ESM-1b/ESM-2.
Prepare csv file with column of sequence data.
This code outputs temporal sequences named 'sequenceTensor1/2/... .pt' to save memory.
The resulting processed sequence (all sequences) is saved in 'sequenceTensor_a**b**gamma**.pt'.
Please remove the temporal files if necessary.
"""

import torch
import csv
import numpy as np
import os

# Check if CUDA is available and print GPU information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# parameters
data_path = '/content/CCS_project_part1/processed_data/immunopeptide_data_JPST002044.tsv'
column_idx = 0              # column index of sequence data in csv file
a = 1000                    # parameter for positional encoding
b = 1                       # parameter for positional encoding
gamma = 0                   # parameter for positional encoding
output_dir = './output'     # directory to save output files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# file open
with open(data_path) as f:
    reader = csv.reader(f, delimiter='\t')
    data = [row for row in reader]
del(data[0])    # remove label, if necessary

print("File opened")

# load ESM model - the t6 version has 6 layers (0-5)
datasize = 20000            # batch size for preprocessing to save memory
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
model = model.to(device)    # Move model to GPU
batch_converter = alphabet.get_batch_converter()
nIteration = len(data)//datasize + 1

print("Pretrained model loaded and moved to", device)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Pre-compute positional encoding tensor (moved to GPU if available)
max_seq_len = max(len(row[column_idx]) for row in data)
pos_enc = torch.zeros(max_seq_len, model.embed_dim).to(device)
for pos in range(pos_enc.size()[0]):
    for i in range(pos_enc.size()[1]):
        if i % 2 == 0:
            pos_enc[pos, i] = (np.sin((pos+1)/(a**(i/pos_enc.size()[1]))))**b + gamma
        else:
            pos_enc[pos, i] = (np.cos((pos+1)/(a**((i-1)/pos_enc.size()[1]))))**b + gamma

# apply positional encoding and save processed sequences
for itr in range(nIteration):
    print(f"Processing batch {itr+1}/{nIteration}")
    
    if itr == nIteration-1:
        datalist = data[itr*datasize:]
    else:
        datalist = data[itr*datasize:(itr+1)*datasize]
    
    seqdata = []
    for i in range(len(datalist)):
        seqdata.append(("protein"+str(i), datalist[i][column_idx]))

    batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)
    batch_tokens = batch_tokens.to(device)  # Move batch to GPU

    # Extract per-residue representations (on GPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[5], return_contacts=False)
    token_representations = results["representations"][5]  # Use layer 5 instead of 33

    # Generate per-sequence representations with positional encoding
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []   # list of tensors
    
    for i, (_, seq) in enumerate(seqdata):
        seq_len = len(seq)
        mid_point = round(seq_len/2)
        
        # Split the representation at the midpoint
        tmp_repre_n = token_representations[i, 1:mid_point+1]
        tmp_repre_c = token_representations[i, mid_point+1:seq_len+1]
        
        # Apply positional encoding
        nseq = torch.mul(tmp_repre_n, pos_enc[:mid_point]).mean(0)
        cseq = torch.mul(tmp_repre_c, torch.flip(pos_enc[:seq_len-mid_point], [0])).mean(0)
        
        # Concatenate N-terminal and C-terminal representations
        combined = torch.cat((nseq, cseq)).cpu()  # Move back to CPU for storage
        sequence_representations.append(combined)
    
    # Save the batch to disk (CPU tensors)
    output_path = os.path.join(output_dir, f'sequenceTensor{itr+1}.pt')
    torch.save(sequence_representations, output_path)
    print(f"Saved batch {itr+1} to {output_path}")
    
    # Clean up GPU memory
    del batch_tokens, token_representations, results
    torch.cuda.empty_cache()

# Combine all batches into the final tensor file
print("Combining all batches...")
sequence_representations = []
for itr in range(nIteration):
    batch_path = os.path.join(output_dir, f'sequenceTensor{itr+1}.pt')
    sequence_representations.extend(torch.load(batch_path))

# Save the combined tensor
final_output = os.path.join(output_dir, f'sequenceTensor_a{a}b{b}gamma{gamma}.pt')
torch.save(sequence_representations, final_output)
print(f"Saved combined tensor to {final_output}")
print("Processing complete!")
