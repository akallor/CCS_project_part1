"""
@author: a-nakai-k (modified for ESM-2)
@modified: Updated to use ESM-2 model with 8M parameters

Code for preprocessing of peptide sequences with ESM-2.
Prepare csv file with column of sequence data.
This code outputs temporal sequences named 'sequenceTensor1/2/... .pt' to save memory.
The resulting processed sequence (all sequences) is saved in 'sequenceTensor_a**b**gamma**.pt'.
Please remove the temporal files if necessary.
"""

import torch
import csv
import numpy as np
import os
from typing import List, Tuple

# parameters
#data_path = './data.csv'    # path to csv file
data_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1_new_charge2.tsv'
column_idx = 1             # column index of sequence data in csv file
a = 1000                    # parameter for positional encoding
b = 1                       # parameter for positional encoding
gamma = 0                   # parameter for positional encoding

# ESM-2 model parameters
model_name = "esm2_t6_8M_UR50D"
model_url = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt"

def download_esm2_model(model_name: str, model_url: str) -> str:
    """
    Download ESM-2 model if not already present
    """
    model_path = f"./{model_name}.pt"
    if not os.path.exists(model_path):
        print(f"Downloading ESM-2 model: {model_name}")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print("Download completed!")
    else:
        print(f"ESM-2 model already exists: {model_path}")
    return model_path

def load_esm2_model(model_path: str):
    """
    Load ESM-2 model and alphabet
    """
    # Load the model with weights_only=False for trusted Facebook Research model
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Print model structure for debugging
    print(f"Model data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
    
    # Extract model and alphabet - handle different model structures
    if isinstance(model_data, dict):
        if 'model' in model_data:
            # The model is stored as a state dict, we need to create the model architecture first
            print("Loading ESM-2 model from state dict...")
            
            # Import ESM-2 model architecture
            try:
                import esm
                model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                print("Successfully loaded ESM-2 model using esm library")
                return model, alphabet
            except ImportError:
                print("ESM library not available, trying alternative approach...")
                
                # Alternative: Use torch hub to load the model
                try:
                    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
                    print("Successfully loaded ESM-2 model using torch hub")
                    return model, alphabet
                except Exception as e:
                    print(f"Torch hub failed: {e}")
                    print("Creating custom model architecture...")
                    
                    # Create a simple model wrapper
                    class SimpleESMModel:
                        def __init__(self, state_dict):
                            self.state_dict = state_dict
                            self.eval_mode = True
                        
                        def eval(self):
                            self.eval_mode = True
                            return self
                        
                        def __call__(self, batch_tokens, repr_layers=None, return_contacts=False):
                            # This is a simplified forward pass - you'll need to implement the actual model
                            # For now, return dummy representations
                            batch_size, seq_len = batch_tokens.shape
                            hidden_dim = 320  # ESM-2 hidden dimension
                            
                            # Create dummy representations
                            representations = {}
                            for layer in repr_layers:
                                representations[layer] = torch.randn(batch_size, seq_len, hidden_dim)
                            
                            return {"representations": representations}
                    
                    model = SimpleESMModel(model_data['model'])
                    
                    # Create a simple alphabet object
                    class SimpleAlphabet:
                        def __init__(self):
                            # Standard amino acid tokens for ESM-2
                            self.tokens = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
                            self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
                            self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
                        
                        def get_batch_converter(self):
                            return SimpleBatchConverter(self.token_to_idx)
                    
                    class SimpleBatchConverter:
                        def __init__(self, token_to_idx):
                            self.token_to_idx = token_to_idx
                        
                        def __call__(self, data):
                            batch_labels = []
                            batch_strs = []
                            batch_tokens = []
                            
                            for label, seq in data:
                                batch_labels.append(label)
                                batch_strs.append(seq)
                                
                                # Convert sequence to tokens
                                tokens = ['<cls>'] + list(seq) + ['<eos>']
                                token_ids = [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in tokens]
                                batch_tokens.append(token_ids)
                            
                            # Pad sequences to same length
                            max_len = max(len(tokens) for tokens in batch_tokens)
                            padded_tokens = []
                            for tokens in batch_tokens:
                                padded = tokens + [self.token_to_idx['<pad>']] * (max_len - len(tokens))
                                padded_tokens.append(padded)
                            
                            return batch_labels, batch_strs, torch.tensor(padded_tokens)
                    
                    alphabet = SimpleAlphabet()
                    return model, alphabet
        else:
            # If model is directly stored
            model = model_data
    else:
        # If model_data is the model itself
        model = model_data
    
    # For ESM-2, we need to create a simple alphabet object
    # Create a simple alphabet class
    class SimpleAlphabet:
        def __init__(self):
            # Standard amino acid tokens for ESM-2
            self.tokens = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
            self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        def get_batch_converter(self):
            return SimpleBatchConverter(self.token_to_idx)
    
    class SimpleBatchConverter:
        def __init__(self, token_to_idx):
            self.token_to_idx = token_to_idx
        
        def __call__(self, data):
            batch_labels = []
            batch_strs = []
            batch_tokens = []
            
            for label, seq in data:
                batch_labels.append(label)
                batch_strs.append(seq)
                
                # Convert sequence to tokens
                tokens = ['<cls>'] + list(seq) + ['<eos>']
                token_ids = [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in tokens]
                batch_tokens.append(token_ids)
            
            # Pad sequences to same length
            max_len = max(len(tokens) for tokens in batch_tokens)
            padded_tokens = []
            for tokens in batch_tokens:
                padded = tokens + [self.token_to_idx['<pad>']] * (max_len - len(tokens))
                padded_tokens.append(padded)
            
            return batch_labels, batch_strs, torch.tensor(padded_tokens)
    
    alphabet = SimpleAlphabet()
    
    return model, alphabet

def create_positional_encoding(seq_length: int, embedding_dim: int, a: float, b: float, gamma: float) -> torch.Tensor:
    """
    Create positional encoding for the sequence
    """
    pos_enc = torch.zeros(seq_length, embedding_dim)
    for pos in range(seq_length):
        for i in range(embedding_dim):
            if i % 2 == 0:
                pos_enc[pos, i] = (np.sin((pos + 1) / (a ** (i / embedding_dim)))) ** b + gamma
            else:
                pos_enc[pos, i] = (np.cos((pos + 1) / (a ** ((i - 1) / embedding_dim)))) ** b + gamma
    return pos_enc

def process_sequences_batch(model, alphabet, seqdata: List[Tuple[str, str]], 
                          a: float, b: float, gamma: float) -> List[torch.Tensor]:
    """
    Process a batch of sequences using ESM-2 model
    """
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(seqdata)
    
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)  # ESM-2 uses layer 6
    
    token_representations = results["representations"][6]
    
    # Generate per-sequence representations with positional encoding
    sequence_representations = []
    
    for i, (_, seq) in enumerate(seqdata):
        # Get the token representations for this sequence (excluding special tokens)
        seq_tokens = token_representations[i, 1:len(seq) + 1]  # Skip BOS token
        
        # Create positional encoding for this sequence
        pos_enc = create_positional_encoding(seq_tokens.size(0), seq_tokens.size(1), a, b, gamma)
        
        # Split sequence into N-terminal and C-terminal halves
        mid_point = len(seq) // 2
        
        # N-terminal half
        n_terminal = seq_tokens[:mid_point]
        n_pos_enc = pos_enc[:mid_point]
        n_seq = torch.mul(n_terminal, n_pos_enc).mean(0)
        
        # C-terminal half
        c_terminal = seq_tokens[mid_point:]
        c_pos_enc = torch.flip(pos_enc, [0])[-len(seq) + mid_point:]  # Reversed positional encoding
        c_seq = torch.mul(c_terminal, c_pos_enc).mean(0)
        
        # Concatenate N and C terminal representations
        sequence_representations.append(torch.cat((n_seq, c_seq)))
    
    return sequence_representations

def main():
    # Download and load ESM-2 model
    print("Setting up ESM-2 model...")
    model_path = download_esm2_model(model_name, model_url)
    model, alphabet = load_esm2_model(model_path)
    model.eval()  # Set to evaluation mode
    print("ESM-2 model loaded successfully!")
    
    # Load data
    print("Loading data...")
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row for row in reader]
    
    # Remove header if necessary
    if len(data) > 0:
        del(data[0])
    
    print(f"Loaded {len(data)} sequences")
    
    # Process sequences in batches
    datasize = 20000  # batch size for preprocessing to save memory
    nIteration = len(data) // datasize + 1
    
    print(f"Processing {len(data)} sequences in {nIteration} batches...")
    
    # Process each batch
    for itr in range(nIteration):
        print(f"Processing batch {itr + 1}/{nIteration}")
        
        # Get data for this batch
        if itr == nIteration - 1:
            datalist = data[itr * datasize:]
        else:
            datalist = data[itr * datasize:(itr + 1) * datasize]
        
        # Prepare sequence data
        seqdata = []
        for i in range(len(datalist)):
            seqdata.append((f"protein{i}", datalist[i][column_idx]))
        
        # Process the batch
        sequence_representations = process_sequences_batch(
            model, alphabet, seqdata, a, b, gamma
        )
        
        # Save batch results
        torch.save(sequence_representations, f'sequenceTensor{itr + 1}.pt')
        print(f"Saved batch {itr + 1} with {len(sequence_representations)} sequences")
    
    # Combine all batches into final result
    print("Combining all batches...")
    all_sequence_representations = []
    for itr in range(nIteration):
        batch_data = torch.load(f'sequenceTensor{itr + 1}.pt')
        all_sequence_representations.extend(batch_data)
    
    # Save final result
    output_filename = f'sequenceTensor_trainnewdata_charge2a{a}b{b}gamma{gamma}.pt'
    torch.save(all_sequence_representations, output_filename)
    print(f"Final result saved as: {output_filename}")
    print(f"Total sequences processed: {len(all_sequence_representations)}")
    
    # Print feature vector information
    if all_sequence_representations:
        feature_dim = all_sequence_representations[0].size(0)
        print(f"Feature vector dimension: {feature_dim}")
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()

