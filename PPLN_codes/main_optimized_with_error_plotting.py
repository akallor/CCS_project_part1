"""
@author: a-nakai-k (original code)
Enhanced version with improved architectures, error logging, and visualization

Code for CCS value prediction using preprocessed sequences with improved model architecture.
Features:
- Multiple model architectures (original, improved, ensemble)
- Comprehensive error logging and metrics tracking
- Training visualization with smoothed curves
- Bias correction and advanced training techniques
"""

import os
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
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

np.set_printoptions(threshold=np.inf)

# parameters
data_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/train_1.tsv'
data_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/test_1.tsv'
column_idx_expccs = 2
column_idx_z = 3
column_idx_mass = 4
sequence_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/sequenceTensor_mhcI_train_a1000b1gamma0.pt'
sequence_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/sequenceTensor_mhcI_test_a1000b1gamma0.pt'
bs = 200
lr_adam = 0.0003
num_ep = 400
istestloss = True

# Model selection: 'original', 'improved', or 'ensemble'
model_type = 'improved'

# ============= DATA LOADING AND PREPARATION =============

def load_data():
    # Load training data
    with open(data_path_train) as f:
        reader = csv.reader(f, delimiter='\t')
        datalist_train = [row for row in reader]
    del(datalist_train[0])  # remove label if necessary

    # Load test data
    with open(data_path_test) as f:
        reader = csv.reader(f, delimiter='\t')
        datalist_test = [row for row in reader]
    del(datalist_test[0])   # remove label if necessary

    # Load sequence representations
    sequence_representations_train = torch.load(sequence_path_train)
    sequence_representations_test = torch.load(sequence_path_test)

    return process_data(datalist_train, datalist_test, sequence_representations_train, sequence_representations_test)

def process_data(datalist_train, datalist_test, sequence_representations_train, sequence_representations_test):
    if istestloss:
        ccs_test = []
    ccs_train = []
    z_test = []
    z_train = []
    seq_testl = []
    seq_trainl = []
    mass_test = []
    mass_train = []

    # Process training data
    for i in range(len(datalist_train)):
        ccs_train.append(float(datalist_train[i][column_idx_expccs]))
        z_train.append(float(datalist_train[i][column_idx_z]))
        mass_train.append(float(datalist_train[i][column_idx_mass]))
        seq_trainl.append(sequence_representations_train[i])

    # Process test data
    for i in range(len(datalist_test)):
        if istestloss:
            ccs_test.append(float(datalist_test[i][column_idx_expccs]))
        z_test.append(float(datalist_test[i][column_idx_z]))
        mass_test.append(float(datalist_test[i][column_idx_mass]))
        seq_testl.append(sequence_representations_test[i])

    # Convert to tensors
    if istestloss:
        ccs_test = torch.tensor(ccs_test)
    ccs_train = torch.tensor(ccs_train)
    z_test = torch.tensor(z_test)
    z_train = torch.tensor(z_train)
    seq_test = torch.stack(seq_testl)
    seq_train = torch.stack(seq_trainl)
    mass_test = torch.tensor(mass_test)
    mass_train = torch.tensor(mass_train)

    # Create datasets
    dataset_train = TensorDataset(z_train, seq_train, mass_train, ccs_train)
    if istestloss:
        dataset_test = TensorDataset(z_test, seq_test, mass_test, ccs_test)
    else:
        dataset_test = TensorDataset(z_test, seq_test, mass_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False)

    return train_loader, test_loader

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

# Improved model architecture
class ImprovedCCSPredictor(nn.Module):
    def __init__(self, esm_dim=1280*2, dropout_rate=0.2):
        super(ImprovedCCSPredictor, self).__init__()
        
        # Sequence processor
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
        
        # Charge/mass processor
        self.charge_mass_processor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Cross-attention
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        # Final predictor
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
        
        # Bias corrector
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
        
        # Process charge and mass
        cm_processed = self.charge_mass_processor(charge_mass)
        
        # Combine features
        combined = torch.cat([seq_processed, cm_processed], dim=1)
        
        # Initial prediction
        initial_pred = self.predictor(combined)
        
        # Bias correction
        bias_correction = self.bias_corrector(initial_pred)
        final_pred = initial_pred + bias_correction
        
        return final_pred

# Ensemble model
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

class BiasAwareLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(BiasAwareLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        residuals = predictions - targets
        sorted_indices = torch.argsort(targets.flatten())
        sorted_residuals = residuals.flatten()[sorted_indices]
        
        n_bins = 10
        bin_size = len(sorted_residuals) // n_bins
        bias_penalty = 0
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_residuals)
            bin_residuals = sorted_residuals[start_idx:end_idx]
            mean_residual = torch.mean(bin_residuals)
            bias_penalty += torch.abs(mean_residual)
        
        return mse_loss + self.alpha * bias_penalty

# ============= TRAINING FUNCTIONS =============

def train_improved(data_loader, model, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item() * num_batch
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(ccs.view(num_batch, -1).detach().cpu().numpy())
        
        if (batch + 1) % 100 == 0:
            current = (batch + 1) * num_batch
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(data_loader.dataset):>5d}]")
    
    avg_loss = total_loss / len(data_loader.dataset)
    rmse = np.sqrt(avg_loss)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    evs = explained_variance_score(all_targets, all_preds)
    
    return rmse, mae, r2, evs

def test(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch, (z, seq, mass, ccs) in enumerate(data_loader):
            z = z.to(device, dtype=torch.float)
            seq = seq.to(device)
            mass = mass.to(device, dtype=torch.float)
            ccs = ccs.to(device, dtype=torch.float)
            
            num_batch = len(z)
            
            # Handle different model types
            if hasattr(model, 'sequence_processor'):  # ImprovedCCSPredictor
                charge_mass = torch.cat((z.view(num_batch,-1), mass.view(num_batch,-1)), dim=1)
                output = model(seq, charge_mass)
            else:  # Original or Ensemble model
                output = model(seq, torch.cat((z.view(num_batch,-1), mass.view(num_batch,-1)), dim=1))
            
            loss = criterion(output, ccs.view(num_batch, -1))
            total_loss += loss.item() * num_batch
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(ccs.view(num_batch, -1).detach().cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    rmse = np.sqrt(avg_loss)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    evs = explained_variance_score(all_targets, all_preds)
    predictCCS = np.array(all_preds)
    
    return rmse, mae, r2, evs, predictCCS

# ============= PLOTTING FUNCTIONS =============

def plot_training_metrics(history_df, plot_dir):
    # Apply smoothing
    sigma = 2
    for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_evs', 'test_evs']:
        history_df[f'{metric}_smooth'] = gaussian_filter1d(history_df[metric], sigma=sigma)

    # Identify best epoch
    best_epoch = history_df['test_rmse'].idxmin()
    best_epoch_num = history_df.loc[best_epoch, 'epoch']

    # Create directory for plots
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
    plt.savefig(f'{plot_dir}/rmse_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig(f'{plot_dir}/mae_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig(f'{plot_dir}/r2_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig(f'{plot_dir}/explained_variance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============= MAIN FUNCTION =============

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

    # Load and prepare data
    train_loader, test_loader = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection
    if model_type == 'original':
        model = CCSpredictor_PretrainedESM().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_adam)
        scheduler = None
        print("Using original CCSpredictor_PretrainedESM model")
        
    elif model_type == 'improved':
        model = ImprovedCCSPredictor().to(device)
        criterion = BiasAwareLoss(alpha=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_ep)
        print("Using improved CCS predictor with bias correction")
        
    elif model_type == 'ensemble':
        model = EnsembleCCSPredictor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_ep)
        print("Using ensemble CCS predictor")

    # Training loop
    for epoch in range(num_ep):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        # Training
        train_rmse, train_mae, train_r2, train_evs = train_improved(
            train_loader, model, optimizer, criterion, device, scheduler)
        
        # Testing
        test_rmse, test_mae, test_r2, test_evs, predictCCS = test(
            test_loader, model, criterion, device)
        
        # Record metrics
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

    # Save model and results
    model_filename = f'trainedmodel_{model_type}.pt'
    torch.save(model.to('cpu').state_dict(), model_filename)
    
    # Save predictions
    output_filename = f"/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/out_test_predictCCS_mhcI_{model_type}.csv"
    np.savetxt(output_filename, predictCCS, delimiter=",")
    
    # Save metrics history
    history_df = pd.DataFrame(history)
    metrics_filename = f"/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/training_metrics_{model_type}.tsv"
    history_df.to_csv(metrics_filename, sep='\t', index=False)
    
    # Plot metrics
    plot_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results"
    plot_training_metrics(history_df, plot_dir)

if __name__ == '__main__':
    main() 
