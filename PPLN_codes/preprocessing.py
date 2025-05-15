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

# parameters
data_path = './data.csv'    # path to csv file
column_idx = 1              # column index of sequence data in csv file
a = 1000                    # parameter for positional encoding
b = 1                       # parameter for positional encoding
gamma = 0                   # parameter for positional encoding

# file open
with open(data_path) as f:
    reader = csv.reader(f)
    data = [row for row in reader]
del(data[0])    # remove label, if necessary

print("File opened")

# load ESM-1b model
datasize = 20000            # batch size for preprocessing to save memory
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")   # not need to clone and/or install ESM
batch_converter = alphabet.get_batch_converter()
nIteration = len(data)//datasize +1

print("Pretrained model loaded")

# apply positional encoding and save processed sequences
for itr in range(nIteration):
    # print(itr)
    if itr==nIteration-1:
        datalist = data[itr*datasize:]
    else:
        datalist = data[itr*datasize:(itr+1)*datasize]
    seqdata = []
    for i in range(len(datalist)):
        seqdata.append(("protein"+str(i), datalist[i][column_idx]))

    batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations with positional encoding
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []   # list of tensors
    pos_enc = torch.zeros(token_representations.size()[1],token_representations.size()[2])
    for pos in range(pos_enc.size()[0]):
        for i in range(pos_enc.size()[1]):
            if i%2 == 0:
                pos_enc[pos,i] = (np.sin((pos+1)/(a**(i/pos_enc.size()[1]))))**b + gamma
            else:
                pos_enc[pos,i] = (np.cos((pos+1)/(a**((i-1)/pos_enc.size()[1]))))**b + gamma
    for i, (_, seq) in enumerate(seqdata):
        tmp_repre_n = token_representations[i, 1 : round(len(seq)/2) + 1]
        tmp_repre_c = token_representations[i, round(len(seq)/2) + 1 : len(seq) + 1]
        nseq = torch.mul(tmp_repre_n,pos_enc[0:round(len(seq)/2),]).mean(0)
        cseq = torch.mul(tmp_repre_c,reversed(pos_enc)[-len(seq)+round(len(seq)/2):,]).mean(0)
        sequence_representations.append(torch.cat((nseq,cseq)))
        # if itr == 1:
        #     if i == 1:
        #         print(token_representations[i,1:len(seq)+1])
        #         print(tmp_repre_n)
        #         print(tmp_repre_c)
        #         print(pos_enc[0:round(len(seq)/2),])
        #         print(reversed(pos_enc)[-len(seq)+round(len(seq)/2):,])
        #         print(torch.cat((nseq,cseq)).size())
    torch.save(sequence_representations, 'sequenceTensor'+str(itr+1)+'.pt')

# create resulting .pt file
sequence_representations = []
for itr in range(nIteration):
    sequence_representations.extend(torch.load('sequenceTensor'+str(itr+1)+'.pt'))
torch.save(sequence_representations, 'sequenceTensor_a'+str(a)+'b'+str(b)+'gamma'+str(gamma)+'.pt')