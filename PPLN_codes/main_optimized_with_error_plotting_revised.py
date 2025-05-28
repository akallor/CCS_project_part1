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
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import math

np.set_printoptions(threshold=np.inf)

# parameters
data_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1.tsv'
data_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1.tsv'
column_idx_expccs = 2
column_idx_z = 3
column_idx_mass = 4
sequence_path_train = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/sequenceTensor_mhcI_train_a1000b1gamma0.pt'
sequence_path_test = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/sequenceTensor_mhcI_test_a1000b1gamma0.pt'
bs = 200
lr_adam = 0.0001
num_ep = 400
istestloss = True

# Training parameters
warmup_epochs = 10
patience = 15
accumulation_steps = 8
ema_decay = 0.999
max_grad_norm = 0.5

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

    # Create normalizer
    normalizer = FeatureNormalizer()
    
    # Prepare training data
    train_seq = torch.stack(sequence_representations_train)
    
    # Convert string data to float tensors
    train_z = torch.tensor([float(row[column_idx_z]) for row in datalist_train])
    train_mass = torch.tensor([float(row[column_idx_mass]) for row in datalist_train])
    train_cm = torch.stack([train_z, train_mass], dim=1)
    train_ccs = torch.tensor([float(row[column_idx_expccs]) for row in datalist_train])
    
    # Fit and transform training data
    normalizer.fit(train_seq, train_cm, train_ccs)
    train_seq_norm, train_cm_norm, train_ccs_norm = normalizer.transform(train_seq, train_cm, train_ccs)
    
    # Transform test data
    test_seq = torch.stack(sequence_representations_test)
    test_z = torch.tensor([float(row[column_idx_z]) for row in datalist_test])
    test_mass = torch.tensor([float(row[column_idx_mass]) for row in datalist_test])
    test_cm = torch.stack([test_z, test_mass], dim=1)
    
    test_seq_norm, test_cm_norm = normalizer.transform(test_seq, test_cm)
    
    if istestloss:
        test_ccs = torch.tensor([float(row[column_idx_expccs]) for row in datalist_test])
        test_ccs_norm = torch.FloatTensor(normalizer.target_scaler.transform(test_ccs.view(-1, 1))).squeeze()
    
    # Create datasets with normalized data
    dataset_train = TensorDataset(train_cm_norm, train_seq_norm, train_ccs_norm)
    if istestloss:
        dataset_test = TensorDataset(test_cm_norm, test_seq_norm, test_ccs_norm)
    else:
        dataset_test = TensorDataset(test_cm_norm, test_seq_norm)
    
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False)
    
    return train_loader, test_loader, normalizer

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
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = F.gelu(self.linear1(x))
        x = self.norm2(x)
        x = self.dropout(self.linear2(x))
        return x + identity

class FeatureNormalizer:
    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.charge_mass_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def fit(self, seq_features, charge_mass, targets):
        self.seq_scaler.fit(seq_features.view(-1, seq_features.size(-1)))
        self.charge_mass_scaler.fit(charge_mass)
        self.target_scaler.fit(targets.reshape(-1, 1))
        
    def transform(self, seq_features, charge_mass, targets=None):
        seq_norm = torch.FloatTensor(self.seq_scaler.transform(seq_features.view(-1, seq_features.size(-1)))).view(seq_features.size())
        cm_norm = torch.FloatTensor(self.charge_mass_scaler.transform(charge_mass))
        if targets is not None:
            targets_norm = torch.FloatTensor(self.target_scaler.transform(targets.reshape(-1, 1))).squeeze()
            return seq_norm, cm_norm, targets_norm
        return seq_norm, cm_norm
        
    def inverse_transform_targets(self, targets):
        return self.target_scaler.inverse_transform(targets.reshape(-1, 1)).squeeze()

class ImprovedCCSPredictor(nn.Module):
    def __init__(self, esm_dim=1280*2, dropout_rate=0.2):
        super(ImprovedCCSPredictor, self).__init__()
        
        self.dropout_rate = dropout_rate
        hidden_dim = 256
        
        # Sequence processor with residual connections
        self.sequence_processor = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim, hidden_dim * 2),
            ResidualBlock(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim)
        )
        
        # Charge/mass processor
        self.charge_mass_processor = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            ResidualBlock(32, 64)
        )
        
        # Final predictor with reduced complexity
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights with a smaller range
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, seq_features, charge_mass):
        # Apply stronger regularization during training
        if self.training:
            seq_features = F.dropout(seq_features, p=0.1, training=True)
            charge_mass = F.dropout(charge_mass, p=0.05, training=True)
        
        seq_processed = self.sequence_processor(seq_features)
        cm_processed = self.charge_mass_processor(charge_mass)
        combined = torch.cat([seq_processed, cm_processed], dim=1)
        return self.predictor(combined)

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

def get_lr_multiplier(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def train_improved(data_loader, model, optimizer, criterion, device, scheduler=None, epoch=0, ema=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    optimizer.zero_grad()
    
    for batch_idx, (z, seq, ccs) in enumerate(data_loader):
        z = z.to(device, dtype=torch.float)
        seq = seq.to(device)
        ccs = ccs.to(device, dtype=torch.float)
        
        num_batch = len(z)
        charge_mass = torch.cat((z.view(num_batch, -1), seq[:, -2:].view(num_batch, -1)), dim=1)
        
        # Forward pass
        output = model(seq, charge_mass)
        loss = criterion(output, ccs.view(num_batch, -1))
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Apply warmup
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_adam * get_lr_multiplier(epoch, warmup_epochs)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA model
            if ema is not None:
                ema.update(model)
        
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item() * num_batch * accumulation_steps
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(ccs.view(num_batch, -1).detach().cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            current = (batch_idx + 1) * num_batch
            print(f"loss: {loss.item() * accumulation_steps:>7f}  [{current:>5d}/{len(data_loader.dataset):>5d}]")
    
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
        for batch, (z, seq, ccs) in enumerate(data_loader):
            z = z.to(device, dtype=torch.float)
            seq = seq.to(device)
            ccs = ccs.to(device, dtype=torch.float)
            
            num_batch = len(z)
            
            # Handle different model types
            if hasattr(model, 'sequence_processor'):  # ImprovedCCSPredictor
                charge_mass = torch.cat((z.view(num_batch,-1), seq[:, -2:].view(num_batch,-1)), dim=1)
                output = model(seq, charge_mass)
            else:  # Original or Ensemble model
                output = model(seq, torch.cat((z.view(num_batch,-1), seq[:, -2:].view(num_batch,-1)), dim=1))
            
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

# EMA Model Averaging
class EMA:
    def __init__(self, model, decay):
        self.model = deepcopy(model)
        self.decay = decay
        self.model.eval()
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

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

    # Load and prepare data with normalization
    train_loader, test_loader, normalizer = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection
    if model_type == 'improved':
        model = ImprovedCCSPredictor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_adam,
            weight_decay=0.1,
            betas=(0.9, 0.999)
        )
        
        # Cosine schedule with warmup
        def get_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        scheduler = get_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_epochs * len(train_loader) // accumulation_steps,
            num_training_steps=num_ep * len(train_loader) // accumulation_steps
        )
        
        ema = EMA(model, decay=ema_decay)
        print("Using improved CCS predictor with enhanced stability")
    
    # Early stopping setup
    best_test_rmse = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_ep):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        # Training
        train_rmse, train_mae, train_r2, train_evs = train_improved(
            train_loader, model, optimizer, criterion, device, scheduler, epoch, ema)
        
        # Testing with EMA model
        test_rmse, test_mae, test_r2, test_evs, predictCCS = test(
            test_loader, ema.model if ema else model, criterion, device)
        
        # Early stopping check
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            patience_counter = 0
            best_model_state = deepcopy(ema.model.state_dict() if ema else model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
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

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model and results
    model_filename = f'trainedmodel_{model_type}.pt'
    torch.save(model.to('cpu').state_dict(), model_filename)
    
    # Save predictions
    predictCCS = normalizer.inverse_transform_targets(torch.tensor(predictCCS))
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
