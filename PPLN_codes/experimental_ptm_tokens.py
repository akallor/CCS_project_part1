import torch
import csv
import re
import numpy as np
from collections import defaultdict

# === Parameters ===
data_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/train_data/train_mhc1.tsv'
column_seq = 1      # index of peptide sequence (e.g., Peptide column)
column_ptm = 2      # index of PTM annotation column (e.g., PTM column)
a, b, gamma = 1000, 1, 0
datasize = 20000

# === Step 1: Read data ===
with open(data_path) as f:
    reader = csv.reader(f, delimiter='\t')
    data = [row for row in reader]
header = data.pop(0)

print("File opened")

# === Step 2: Extract all unique PTMs from peptide column and map to names from PTM column ===
ptm_mass_to_name = {}
for row in data:
    mod_seq = row[column_seq]
    ptm_ann = row[column_ptm]
    masses = re.findall(r'\((\+\d+\.\d+)\)', mod_seq)
    ptm_name_list = [x.strip() for x in ptm_ann.split(';')]
    for m, ptm_name in zip(masses, ptm_name_list):
        ptm_mass_to_name[m] = ptm_name

# === Step 3: Assign embeddings to all unique PTM names ===
all_ptm_names = sorted(set(ptm_mass_to_name.values()))
ptm_name_to_embedding = {
    ptm_name: torch.nn.functional.one_hot(torch.tensor(i), num_classes=len(all_ptm_names)).float()
    for i, ptm_name in enumerate(all_ptm_names)
}
ptm_name_to_embedding['None'] = torch.zeros(len(all_ptm_names))

def get_combined_ptm_embedding(mod_seq, ptm_ann):
    masses = re.findall(r'\((\+\d+\.\d+)\)', mod_seq)
    ptm_name_list = [x.strip() for x in ptm_ann.split(';')]

    # fallback if there's a mismatch in annotation length
    if len(masses) != len(ptm_name_list):
        return ptm_name_to_embedding['None']

    ptm_embeddings = []
    for mass in masses:
        ptm_name = ptm_mass_to_name.get(mass, 'None')
        ptm_embeddings.append(ptm_name_to_embedding.get(ptm_name, ptm_name_to_embedding['None']))

    if not ptm_embeddings:
        return ptm_name_to_embedding['None']

    return torch.stack(ptm_embeddings).mean(dim=0)  # mean of all PTM embeddings

print("PTM mappings established:")
for k, v in ptm_mass_to_name.items():
    print(f"{k} → {v}")

# === Step 4: Load pretrained ESM-1b ===
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
batch_converter = alphabet.get_batch_converter()
nIteration = len(data) // datasize + 1

print("Pretrained model loaded")

# === Step 5: Process data in batches and apply positional + PTM encoding ===
for itr in range(nIteration):
    if itr == nIteration - 1:
        datalist = data[itr * datasize:]
    else:
        datalist = data[itr * datasize:(itr + 1) * datasize]

    seqdata = []
    ptm_embeddings = []
    for i, row in enumerate(datalist):
        mod_seq = row[column_seq]
        ptm_ann = row[column_ptm]
        clean_seq = re.sub(r'\(\+\d+\.\d+\)', '', mod_seq)  # remove PTM mass from sequence
        seqdata.append(("protein" + str(i), clean_seq))
        ptm_embeddings.append(get_combined_ptm_embedding(mod_seq, ptm_ann))

    batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)

    # Extract ESM representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Positional encoding
    sequence_representations = []
    pos_enc = torch.zeros(token_representations.size()[1], token_representations.size()[2])
    for pos in range(pos_enc.size()[0]):
        for i in range(pos_enc.size()[1]):
            if i % 2 == 0:
                pos_enc[pos, i] = (np.sin((pos + 1) / (a ** (i / pos_enc.size()[1])))) ** b + gamma
            else:
                pos_enc[pos, i] = (np.cos((pos + 1) / (a ** ((i - 1) / pos_enc.size()[1])))) ** b + gamma

    for i, (_, seq) in enumerate(seqdata):
        seq_len = len(seq)
        tmp_repre_n = token_representations[i, 1:round(seq_len / 2) + 1]
        tmp_repre_c = token_representations[i, round(seq_len / 2) + 1:seq_len + 1]
        nseq = torch.mul(tmp_repre_n, pos_enc[0:round(seq_len / 2),]).mean(0)
        cseq = torch.mul(tmp_repre_c, torch.flip(pos_enc, [0])[-seq_len + round(seq_len / 2):,]).mean(0)
        esm_vector = torch.cat((nseq, cseq))  # ESM representation
        final_vec = torch.cat((esm_vector, ptm_embeddings[i]))  # Add PTM embedding
        sequence_representations.append(final_vec)

    torch.save(sequence_representations, f'sequenceTensor{itr + 1}.pt')
    print(f"Saved sequenceTensor{itr + 1}.pt")

# === Step 6: Merge all batch tensors into final file ===
all_sequence_representations = []
for itr in range(nIteration):
    all_sequence_representations.extend(torch.load(f'sequenceTensor{itr + 1}.pt'))

torch.save(all_sequence_representations, f'sequenceTensor_mhc1_train_a{a}b{b}gamma{gamma}.pt')
print(f"Saved final tensor: sequenceTensor_mhc1_train_a{a}b{b}gamma{gamma}.pt")
