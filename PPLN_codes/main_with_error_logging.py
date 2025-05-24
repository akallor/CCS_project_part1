"""
@author: a-nakai-k

Code for CCS value prediction using preprocessed sequences.
The predicted CCS values for test data is saved as csv file named 'out_test_predictCCS.csv'.
The trained model composed of fully connected layers is saved as 'trainedmodel.pt'.
List 'ccstest' in l.49 is experimental CCS values for test data 
and used only for calculating test loss, not used for CCS prediction.
If you do not have experimental CCS values for test data, 
please set the variable 'istestloss' to be 'False'.
"""

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

np.set_printoptions(threshold=np.inf)

# parameters
data_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_2/processed_data/train_1.tsv'    # path to csv file for training
data_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_2/processed_data/test_1.tsv'      # path to csv file for test
column_idx_expccs = 2                   # column index of experimental ccs value data in csv file
column_idx_z = 3                        # column index of charge data in csv file
column_idx_mass = 4                     # column index of mass data in csv file
sequence_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_2/results/sequenceTensor_mhcII_train_a1000b1gamma0.pt' # path to preprocessed sequence data for training
sequence_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_2/results/sequenceTensor_mhcII_test_a1000b1gamma0.pt'   # path to preprocessed sequence data for test
bs = 200                                # batch size
lr_adam = 0.0003                        # learning rate
num_ep = 400                            # number of epochs
istestloss = True

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

# model
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

# Enhanced training function with multiple metrics
def train(data_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    for z, seq, mass, ccs in data_loader:
        z = z.to(device, dtype=torch.float)
        seq = seq.to(device)
        mass = mass.to(device, dtype=torch.float)
        ccs = ccs.to(device, dtype=torch.float)
        optimizer.zero_grad()
        num_batch = len(z)
        output = model(seq, torch.cat((z.view(num_batch, -1), mass.view(num_batch, -1)), dim=1))
        loss = criterion(output, ccs.view(num_batch, -1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_batch
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(ccs.view(num_batch, -1).detach().cpu().numpy())
    avg_loss = total_loss / len(data_loader.dataset)
    rmse = np.sqrt(avg_loss)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    evs = explained_variance_score(all_targets, all_preds)
    return rmse, mae, r2, evs

# Enhanced testing function with multiple metrics
def test(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for z, seq, mass, ccs in data_loader:
            z = z.to(device, dtype=torch.float)
            seq = seq.to(device)
            mass = mass.to(device, dtype=torch.float)
            ccs = ccs.to(device, dtype=torch.float)
            num_batch = len(z)
            output = model(seq, torch.cat((z.view(num_batch, -1), mass.view(num_batch, -1)), dim=1))
            loss = criterion(output, ccs.view(num_batch, -1))
            total_loss += loss.item() * num_batch
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(ccs.view(num_batch, -1).detach().cpu().numpy())
    avg_loss = total_loss / len(data_loader.dataset)
    rmse = np.sqrt(avg_loss)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    evs = explained_variance_score(all_targets, all_preds)
    return rmse, mae, r2, evs, all_preds

# Enhanced main function with comprehensive metrics tracking
def main():
    history = {
        'epoch': [],
        'train_rmse': [],
        'train_mae': [],
        'train_r2': [],
        'train_evs': [],
        'test_rmse': [],
        'test_mae': [],
        'test_r2': [],
        'test_evs': []
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CCSpredictor_PretrainedESM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_adam)
    
    for epoch in range(num_ep):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_rmse, train_mae, train_r2, train_evs = train(train_loader, model, optimizer, criterion, device)
        test_rmse, test_mae, test_r2, test_evs, predictCCS = test(test_loader, model, criterion, device)
        
        history['epoch'].append(epoch + 1)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)
        history['train_evs'].append(train_evs)
        history['test_rmse'].append(test_rmse)
        history['test_mae'].append(test_mae)
        history['test_r2'].append(test_r2)
        history['test_evs'].append(test_evs)
        
        print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}, EVS: {train_evs:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, EVS: {test_evs:.4f}")
    
    torch.save(model.to('cpu').state_dict(), 'trainedmodel.pt')
    np.savetxt("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/out_test_predictCCS_mhcI.csv", predictCCS, delimiter=",")
    pd.DataFrame(history).to_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/training_metrics.tsv", sep='\t', index=False)

# Plotting function to visualize training metrics
def plot_training_metrics():
    # Load the logged metrics
    history_df = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/training_metrics.tsv", sep='\t')

    # Apply smoothing
    sigma = 2  # adjust for smoothing level
    for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_evs', 'test_evs']:
        history_df[f'{metric}_smooth'] = gaussian_filter1d(history_df[metric], sigma=sigma)

    # Identify best epoch (lowest test RMSE)
    best_epoch = history_df['test_rmse'].idxmin()
    best_epoch_num = history_df.loc[best_epoch, 'epoch']

    # Create directory for plots
    plot_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/"
    os.makedirs(plot_dir, exist_ok=True)

    # RMSE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_rmse_smooth'], label='Train RMSE')
    plt.plot(history_df['epoch'], history_df['test_rmse_smooth'], label='Test RMSE')
    plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}rmse_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_mae_smooth'], label='Train MAE')
    plt.plot(history_df['epoch'], history_df['test_mae_smooth'], label='Test MAE')
    plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}mae_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # R² Score Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_r2_smooth'], label='Train R²')
    plt.plot(history_df['epoch'], history_df['test_r2_smooth'], label='Test R²')
    plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('R² over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}r2_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Explained Variance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_evs_smooth'], label='Train Explained Variance')
    plt.plot(history_df['epoch'], history_df['test_evs_smooth'], label='Test Explained Variance')
    plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}explained_variance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Run training
    main()
    
    # Plot results
    plot_training_metrics()
