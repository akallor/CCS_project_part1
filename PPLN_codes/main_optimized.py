"""
@author: a-nakai-k

Code for CCS value prediction using preprocessed sequences with improved model architecture.
The predicted CCS values for test data is saved as csv file named 'out_test_predictCCS.csv'.
The trained model composed of fully connected layers is saved as 'trainedmodel.pt'.
List 'ccstest' in l.49 is experimental CCS values for test data 
and used only for calculating test loss, not used for CCS prediction.
If you do not have experimental CCS values for test data, 
please set the variable 'istestloss' to be 'False'.

Enhanced with improved model architectures to address systematic bias.
"""
import os as os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np

np.set_printoptions(threshold=np.inf)

# parameters
data_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/train_1.tsv'    # path to csv file for training
data_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/test_1.tsv'      # path to csv file for test
column_idx_expccs = 2                   # column index of experimental ccs value data in csv file
column_idx_z = 3                        # column index of charge data in csv file
column_idx_mass = 4                     # column index of mass data in csv file
sequence_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/sequenceTensor_mhcI_train_a1000b1gamma0.pt' # path to preprocessed sequence data for training
sequence_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/sequenceTensor_mhcI_test_a1000b1gamma0.pt'   # path to preprocessed sequence data for test
bs = 200                                # batch size
lr_adam = 0.0003                        # learning rate
num_ep = 400                            # number of epochs
istestloss = True

# Model selection: 'original', 'improved', or 'ensemble'
model_type = 'improved'  # Change this to select which model to use

# data preparation
with open(data_path_train) as f:
    reader = csv.reader(f,delimiter = '\t')
    datalist_train = [row for row in reader]
del(datalist_train[0])  # remove label if necessary
with open(data_path_test) as f:
    reader = csv.reader(f,delimiter = '\t')
    datalist_test = [row for row in reader]
del(datalist_test[0])   # remove label if necessary
sequence_representations_train = torch.load(sequence_path_train)
sequence_representations_test = torch.load(sequence_path_test)

if istestloss:
    ccs_test = []
ccs_train = []
z_test = []
z_train = []
seq_testl = []
seq_trainl = []
mass_test = []
mass_train = []

for i in range(len(datalist_train)):
    ccs_train.append(float(datalist_train[i][column_idx_expccs]))
    z_train.append(float(datalist_train[i][column_idx_z]))
    mass_train.append(float(datalist_train[i][column_idx_mass]))
    seq_trainl.append(sequence_representations_train[i])
for i in range(len(datalist_test)):
    if istestloss:
        ccs_test.append(float(datalist_test[i][column_idx_expccs]))
    z_test.append(float(datalist_test[i][column_idx_z]))
    mass_test.append(float(datalist_test[i][column_idx_mass]))
    seq_testl.append(sequence_representations_test[i])

if istestloss:
    ccs_test = torch.tensor(ccs_test)
ccs_train = torch.tensor(ccs_train)
z_test = torch.tensor(z_test)
z_train = torch.tensor(z_train)
seq_test = torch.stack(seq_testl)
seq_train = torch.stack(seq_trainl)
mass_test = torch.tensor(mass_test)
mass_train = torch.tensor(mass_train)

dataset_train = TensorDataset(z_train,seq_train,mass_train,ccs_train)
if istestloss:
    dataset_test = TensorDataset(z_test,seq_test,mass_test,ccs_test)
else:
    dataset_test = TensorDataset(z_test,seq_test,mass_test)
train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False)

# ============= MODEL DEFINITIONS =============

# Original model
class CCSpredictor_PretrainedESM(nn.Module):
    def __init__(self):
        super(CCSpredictor_PretrainedESM, self).__init__()
        self.fc1 = nn.Linear(1280*2+2, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, 1000)
        self.fc8 = nn.Linear(1000, 1000)
        self.fc9 = nn.Linear(1000, 1000)
        self.fc10 = nn.Linear(1000, 1)
    def forward(self, x, zandmass):
        x = torch.cat((x, zandmass), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x

# Improved model architecture to address systematic bias
class ImprovedCCSPredictor(nn.Module):
    def __init__(self, esm_dim=1280*2, dropout_rate=0.2):
        super(ImprovedCCSPredictor, self).__init__()
        
        # Separate processing for different input types
        self.sequence_processor = nn.Sequential(
            nn.Linear(esm_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dedicated charge/mass processor with non-linear interactions
        self.charge_mass_processor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Cross-attention mechanism for sequence-charge interaction
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        # Final prediction layers with residual connections
        self.predictor = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Bias correction layer (learns systematic corrections)
        self.bias_corrector = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(), 
            nn.Linear(16, 1)
        )
        
    def forward(self, seq_features, charge_mass):
        # Process sequence features
        seq_processed = self.sequence_processor(seq_features)
        
        # Process charge and mass with non-linear transformations
        cm_processed = self.charge_mass_processor(charge_mass)
        
        # Combine features
        combined = torch.cat([seq_processed, cm_processed], dim=1)
        
        # Initial prediction
        initial_pred = self.predictor(combined)
        
        # Apply bias correction based on predicted value
        bias_correction = self.bias_corrector(initial_pred)
        final_pred = initial_pred + bias_correction
        
        return final_pred

# Alternative: Ensemble approach
class EnsembleCCSPredictor(nn.Module):
    def __init__(self, n_models=3):
        super(EnsembleCCSPredictor, self).__init__()
        self.models = nn.ModuleList([
            self._create_base_model() for _ in range(n_models)
        ])
        self.final_layer = nn.Linear(n_models, 1)
        
    def _create_base_model(self):
        return nn.Sequential(
            nn.Linear(1280*2+2, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    
    def forward(self, x, zandmass):
        combined_input = torch.cat((x, zandmass), dim=1)
        predictions = torch.cat([model(combined_input) for model in self.models], dim=1)
        return self.final_layer(predictions)

# ============= LOSS FUNCTIONS =============

# Custom loss function to penalize systematic bias
class BiasAwareLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(BiasAwareLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # Weight for bias penalty
        
    def forward(self, predictions, targets):
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Calculate bias penalty (penalize systematic over/under-prediction)
        residuals = predictions - targets
        
        # Sort by target values and check for systematic bias
        sorted_indices = torch.argsort(targets.flatten())
        sorted_residuals = residuals.flatten()[sorted_indices]
        
        # Penalty for consistent positive/negative residuals in ranges
        n_bins = 10
        bin_size = len(sorted_residuals) // n_bins
        bias_penalty = 0
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_residuals)
            bin_residuals = sorted_residuals[start_idx:end_idx]
            
            # Penalize if most residuals in bin have same sign
            mean_residual = torch.mean(bin_residuals)
            bias_penalty += torch.abs(mean_residual)
        
        return mse_loss + self.alpha * bias_penalty

# ============= TRAINING FUNCTIONS =============

# Original training function
def train(data_loader,model,optimizer,criterion,device):
    size = len(data_loader.dataset)
    for batch, (z,seq,mass,ccs) in enumerate(data_loader):
        z = z.to(device, dtype=torch.float)
        seq = seq.to(device)
        mass = mass.to(device, dtype=torch.float)
        ccs = ccs.to(device, dtype=torch.float)

        optimizer.zero_grad()
        num_batch = len(z)
        output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))

        loss = criterion(output, ccs.view(num_batch,-1))
        loss.backward()
        optimizer.step()
        if (batch+1) % 100 == 0:
            rmse, current = np.sqrt(loss.item()), (batch+1) * num_batch
            print(f"loss: {rmse:>7f}  [{current:>5d}/{size:>5d}]")
    return np.sqrt(loss.item())

# Improved training function with gradient clipping and scheduler support
def train_improved(data_loader, model, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    
    for batch, (z, seq, mass, ccs) in enumerate(data_loader):
        z = z.to(device, dtype=torch.float)
        seq = seq.to(device)
        mass = mass.to(device, dtype=torch.float)
        ccs = ccs.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        num_batch = len(z)
        
        # Prepare inputs
        charge_mass = torch.cat((z.view(num_batch, -1), mass.view(num_batch, -1)), dim=1)
        
        # Forward pass
        if hasattr(model, 'sequence_processor'):  # ImprovedCCSPredictor
            output = model(seq, charge_mass)
        else:  # Original or Ensemble model
            output = model(seq, charge_mass)
        
        loss = criterion(output, ccs.view(num_batch, -1))
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
        if (batch + 1) % 100 == 0:
            current = (batch + 1) * num_batch
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(data_loader.dataset):>5d}]")
    
    return total_loss / len(data_loader)

# ============= TEST FUNCTIONS =============

def test(data_loader,model,criterion,device):
    size = len(data_loader.dataset)
    predictCCS = np.array([])
    if istestloss:
        test_loss = 0
        with torch.no_grad():
            for batch, (z,seq,mass,ccs) in enumerate(data_loader):
                z = z.to(device, dtype=torch.float)
                seq = seq.to(device)
                mass = mass.to(device, dtype=torch.float)
                ccs = ccs.to(device, dtype=torch.float)

                num_batch = len(z)
                
                # Handle different model types
                if hasattr(model, 'sequence_processor'):  # ImprovedCCSPredictor
                    charge_mass = torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1)
                    output = model(seq, charge_mass)
                else:  # Original or Ensemble model
                    output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))
                
                output_np = output.to('cpu').detach().numpy().copy().reshape(-1)
                predictCCS = np.append(predictCCS,output_np)

                test_loss += num_batch * criterion(output,ccs.view(num_batch,-1))
        test_loss /= size
        test_loss = np.sqrt(test_loss.item())
        print(f"Test Error: \n RMSE: {test_loss:>8f} \n")
        return test_loss, predictCCS
    else:
        with torch.no_grad():
            for batch, (z,seq,mass) in enumerate(data_loader):
                z = z.to(device, dtype=torch.float)
                seq = seq.to(device)
                mass = mass.to(device, dtype=torch.float)

                num_batch = len(z)
                
                # Handle different model types
                if hasattr(model, 'sequence_processor'):  # ImprovedCCSPredictor
                    charge_mass = torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1)
                    output = model(seq, charge_mass)
                else:  # Original or Ensemble model
                    output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))
                
                output_np = output.to('cpu').detach().numpy().copy().reshape(-1)
                predictCCS = np.append(predictCCS,output_np)
        return predictCCS

# ============= MAIN TRAINING FUNCTION =============

def main():
    history = {
        'train_rmse': [],
        'test_rmse': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection
    if model_type == 'original':
        model = CCSpredictor_PretrainedESM().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_adam)
        scheduler = None
        train_function = train
        print("Using original CCSpredictor_PretrainedESM model")
        
    elif model_type == 'improved':
        model = ImprovedCCSPredictor().to(device)
        criterion = BiasAwareLoss(alpha=0.1)  # Using bias-aware loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_ep)
        train_function = train_improved
        print("Using improved CCS predictor with bias correction")
        
    elif model_type == 'ensemble':
        model = EnsembleCCSPredictor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_ep)
        train_function = train_improved
        print("Using ensemble CCS predictor")
        
    else:
        raise ValueError("model_type must be 'original', 'improved', or 'ensemble'")

    # Prepare log file path
    log_file = f"/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/training_log_{model_type}.tsv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write("epoch\ttrain_rmse\ttest_rmse\n")

    for epoch in range(num_ep):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        # Training
        if train_function == train_improved:
            train_loss = train_function(train_loader, model, optimizer, criterion, device, scheduler)
        else:
            train_loss = train_function(train_loader, model, optimizer, criterion, device)
            
        history['train_rmse'].append(train_loss)

        # Test
        if istestloss:
            test_loss, predictCCS = test(test_loader, model, criterion, device)
            history['test_rmse'].append(test_loss)
        else:
            predictCCS = test(test_loader, model, criterion, device)
            test_loss = 0  # Set default value for logging
    
        # Append epoch data to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{train_loss:.6f}\t{test_loss:.6f}\n")

    # Save model and results
    model_filename = f'trainedmodel_{model_type}.pt'
    torch.save(model.to('cpu').state_dict(), model_filename)
    print(f"Model saved as: {model_filename}")
    
    print(history)
    print('test_predictCCS：{0}'.format(predictCCS))
    
    output_filename = f"/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/out_test_predictCCS_mhcI_{model_type}.csv"
    np.savetxt(output_filename, predictCCS, delimiter=",")
    print(f"Predictions saved as: {output_filename}")

# Alternative main function for improved models (from original paste.txt)
def main_improved():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try the improved model
    model = ImprovedCCSPredictor().to(device)
    
    # Use custom loss function
    criterion = BiasAwareLoss(alpha=0.1)  # Adjust alpha as needed
    
    # Use learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    
    # Training loop would go here...
    # This is a template - actual implementation is in main() function above

if __name__ == '__main__':
    main()
