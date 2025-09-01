"""
@author: a-nakai-k (modified for ESM-2)
Enhanced version with improved architectures, error logging, and visualization

Code for CCS value prediction using preprocessed sequences with ESM-2 features.
Modified to work with ESM-2 feature extraction output.
"""

import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import math
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.nn.modules.loss import _Loss
import seaborn as sns

np.set_printoptions(threshold=np.inf)

# Data paths - Updated for ESM-2
DATA_PATHS = {
    'train_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1_new_charge2.tsv',
    'test_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1_new_charge2.tsv',
    'train_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_trainnewdata_charge2a1000b1gamma0.pt',  # Updated for ESM-2
    'test_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_testnewdata_charge2a1000b1gamma0.pt',  # Updated for ESM-2
    'results': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results_esm2'
}

class TrainingConfig:
    def __init__(self, model_type='both'):
        # TPU-optimized parameters
        self.bs = 256  # Further reduced batch size
        self.base_lr = 1e-4  # Reduced learning rate
        self.num_epochs = 400
        self.warmup_epochs = 5
        self.patience = 15
        self.accumulation_steps = 1
        
        # Model parameters
        self.model_type = model_type  # 'improved', 'ensemble', or 'both'
        self.ensemble_size = 3
        self.ensemble_weights = None
        self.temperature = 1.0  # Temperature for ensemble model
        self.combined_weights = {'improved': 0.5, 'ensemble': 0.5}  # Weights for 'both' model type
        
        # Training parameters
        self.weight_decay = 1e-4  # Increased weight decay
        self.label_smoothing = 0.0  # Removed label smoothing
        self.dropout_rate = 0.2  # Increased dropout
        self.hidden_dim = 128  # Increased hidden dimension
        self.num_folds = 5
        
        # Learning rate schedule
        self.min_lr = 1e-6
        self.cycle_momentum = False
        self.cycle_decay = 0.95
        
        # Regularization
        self.mixup_alpha = 0.0  # Removed mixup
        self.gradient_clip_val = 0.5  # Reduced gradient clipping

class FeatureNormalizer:
    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.charge_mass_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def fit(self, seq_features, charge_mass, targets):
        # Reshape for proper scaling
        seq_features_2d = seq_features.view(-1, seq_features.size(-1))
        self.seq_scaler.fit(seq_features_2d)
        self.charge_mass_scaler.fit(charge_mass)
        self.target_scaler.fit(targets.reshape(-1, 1))
        
        # Store mean and std for debugging
        self.seq_mean = torch.tensor(self.seq_scaler.mean_, dtype=torch.float32)
        self.seq_std = torch.tensor(self.seq_scaler.scale_, dtype=torch.float32)
        self.target_mean = self.target_scaler.mean_[0]
        self.target_std = self.target_scaler.scale_[0]
        
        print(f"Target mean: {self.target_mean:.4f}, std: {self.target_std:.4f}")
        print(f"Sequence feature dimension: {seq_features.size(-1)}")
        
    def transform(self, seq_features, charge_mass, targets=None):
        # Reshape for proper scaling
        seq_features_2d = seq_features.view(-1, seq_features.size(-1))
        seq_norm = torch.FloatTensor(self.seq_scaler.transform(seq_features_2d)).view(seq_features.size())
        cm_norm = torch.FloatTensor(self.charge_mass_scaler.transform(charge_mass))
        
        if targets is not None:
            targets_norm = torch.FloatTensor(self.target_scaler.transform(targets.reshape(-1, 1))).squeeze()
            return seq_norm, cm_norm, targets_norm
        return seq_norm, cm_norm
        
    def inverse_transform_targets(self, targets):
        # Ensure input is the right shape for inverse transform
        if isinstance(targets, torch.Tensor):
            targets = targets.reshape(-1, 1)
            if targets.device.type != 'cpu':
                targets = targets.cpu()
            targets = targets.numpy()
        else:
            targets = targets.reshape(-1, 1)
            
        # Inverse transform
        return torch.FloatTensor(self.target_scaler.inverse_transform(targets))

class HuberMSELoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberMSELoss, self).__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        diff = pred - target
        mask = torch.abs(diff) <= self.delta
        loss = torch.where(mask,
                          0.5 * diff ** 2,
                          self.delta * torch.abs(diff) - 0.5 * self.delta ** 2)
        return loss.mean()

# Enhanced ResidualBlock with Squeeze-and-Excitation
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_rate=0.1):
        super(EnhancedResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SEBlock(in_features)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.gelu(self.linear1(x))
        x = self.norm2(x)
        x = self.dropout(self.linear2(x))
        x = x.unsqueeze(-1)
        x = self.se(x)
        x = x.squeeze(-1)
        return x + identity

# Improved model architecture - Updated for ESM-2 feature dimensions
class ImprovedCCSPredictor(nn.Module):
    def __init__(self, config, esm_dim=320):  # Updated for ESM-2 (160*2 for N/C split)
        super(ImprovedCCSPredictor, self).__init__()
        
        self.dropout_rate = config.dropout_rate
        hidden_dim = config.hidden_dim
        
        # Input normalization layers
        self.seq_norm = nn.LayerNorm(esm_dim)
        self.cm_norm = nn.LayerNorm(2)
        
        # Sequence processor
        self.sequence_processor = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Charge/mass processor
        self.charge_mass_processor = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Final predictor with residual connections
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights carefully
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, seq_features, charge_mass):
        # Apply input normalization
        seq_features = self.seq_norm(seq_features)
        charge_mass = self.cm_norm(charge_mass)
        
        # Process features
        seq_processed = self.sequence_processor(seq_features)
        cm_processed = self.charge_mass_processor(charge_mass)
        
        # Combine features
        x = torch.cat([seq_processed, cm_processed], dim=1)
        
        # Predict
        return self.predictor(x)

class EnhancedEnsembleCCSPredictor(nn.Module):
    def __init__(self, config, esm_dim=320):  # Updated for ESM-2
        super(EnhancedEnsembleCCSPredictor, self).__init__()
        self.models = nn.ModuleList([
            ImprovedCCSPredictor(config, esm_dim) 
            for _ in range(config.ensemble_size)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(config.ensemble_size) / config.ensemble_size
        )
        self.temperature = config.temperature if hasattr(config, 'temperature') else 1.0
    
    def forward(self, seq_features, charge_mass):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(seq_features, charge_mass)
            predictions.append(pred)
        
        # Stack predictions and apply temperature scaling
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, n_models, 1]
        scaled_weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Weighted average of predictions
        weighted_pred = torch.sum(stacked_preds * scaled_weights.view(1, -1, 1), dim=1)
        
        if self.training:
            return weighted_pred, predictions
        return weighted_pred
    
    def get_diversity_loss(self, predictions):
        # Calculate diversity loss to encourage model disagreement
        predictions = torch.stack(predictions, dim=1)
        mean_pred = torch.mean(predictions, dim=1, keepdim=True)
        # Use L1 loss for diversity and normalize by batch size
        diversity_loss = -torch.mean(torch.abs(predictions - mean_pred)) / predictions.size(0)
        return diversity_loss

def create_model(config, device, esm_dim=320):  # Updated for ESM-2
    if config.model_type == 'improved':
        return ImprovedCCSPredictor(config, esm_dim).to(device)
    elif config.model_type == 'ensemble':
        return EnhancedEnsembleCCSPredictor(config, esm_dim).to(device)
    elif config.model_type == 'both':
        improved_model = ImprovedCCSPredictor(config, esm_dim).to(device)
        ensemble_model = EnhancedEnsembleCCSPredictor(config, esm_dim).to(device)
        # Store config in models for combined prediction weights
        improved_model.config = config
        ensemble_model.config = config
        return {
            'improved': improved_model,
            'ensemble': ensemble_model
        }
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler, epoch):
    if isinstance(model, dict):  # Handle 'both' model type
        train_metrics = {}
        for model_key in ['improved', 'ensemble']:
            model[model_key].train()
            metrics = train_single_model(
                model[model_key],
                train_loader,
                optimizer[model_key],
                criterion,
                device,
                config,
                scheduler[model_key],
                epoch,
                model_type=model_key
            )
            train_metrics[model_key] = metrics
        return train_metrics
    else:
        return train_single_model(model, train_loader, optimizer, criterion, device, config, scheduler, epoch)

def train_single_model(model, train_loader, optimizer, criterion, device, config, scheduler, epoch, model_type=None):
    model.train()
    total_loss = 0
    running_loss = 0.0
    batch_count = 0
    
    # Store predictions and targets for metrics calculation
    epoch_preds = []
    epoch_targets = []
    
    for batch_idx, (charge_mass, seq, ccs) in enumerate(train_loader):
        charge_mass = charge_mass.to(device, dtype=torch.float)
        seq = seq.to(device)
        ccs = ccs.to(device, dtype=torch.float)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'ensemble' or (hasattr(model, '__class__') and model.__class__.__name__ == 'EnhancedEnsembleCCSPredictor'):
            output, ensemble_preds = model(seq, charge_mass)
            # Calculate main loss
            main_loss = criterion(output, ccs.view(-1, 1))
            # Add scaled diversity loss
            if epoch < 5:  # Start with no diversity loss
                diversity_weight = 0.0
            else:  # Gradually increase diversity weight
                diversity_weight = min(0.01, 0.002 * (epoch - 4))  # Cap at 0.01
            diversity_loss = model.get_diversity_loss(ensemble_preds)
            loss = main_loss + diversity_weight * diversity_loss
            
            if batch_idx % 10 == 0:  # Log components every 10 batches
                print(f"\nLoss components for {model_type if model_type else 'ensemble'} - "
                      f"Main: {main_loss:.4f}, Diversity: {diversity_loss:.4f} "
                      f"(weight: {diversity_weight:.4f})")
            
            output_for_metrics = output
        else:
            output = model(seq, charge_mass)
            loss = criterion(output, ccs.view(-1, 1))
            output_for_metrics = output
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
        else:
            print(f"NaN loss detected at batch {batch_idx}")
            continue
        
        # Record metrics
        total_loss += loss.item() * len(charge_mass)
        batch_count += 1
        
        # Store predictions and targets for metrics
        epoch_preds.append(output_for_metrics.detach())
        epoch_targets.append(ccs.view(-1, 1).detach())
        
        # Monitor running loss
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            model_name = model_type if model_type else ('ensemble' if hasattr(model, 'ensemble_weights') else 'improved')
            print(f'Batch {batch_idx + 1}, Running Loss ({model_name}): {running_loss / 10:.4f}')
            running_loss = 0.0
    
    # Calculate final metrics
    with torch.no_grad():
        all_preds = torch.cat(epoch_preds, dim=0).cpu().numpy()
        all_targets = torch.cat(epoch_targets, dim=0).cpu().numpy()
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(train_loader.dataset)
    
    return metrics

def validate(model, val_loader, criterion, device, save_predictions=False, save_path=None, normalizer=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for charge_mass, seq, ccs in val_loader:
            charge_mass = charge_mass.to(device, dtype=torch.float)
            seq = seq.to(device)
            ccs = ccs.to(device, dtype=torch.float)
            
            # Handle ensemble model output
            output = model(seq, charge_mass)
            if isinstance(output, tuple):
                output = output[0]  # Take only the final predictions
            
            loss = criterion(output, ccs.view(-1, 1))
            
            total_loss += loss.item() * len(charge_mass)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(ccs.view(-1, 1).cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Inverse transform predictions if normalizer is provided
    if normalizer is not None:
        # Convert numpy arrays to torch tensors for inverse transform
        all_preds_tensor = torch.from_numpy(all_preds)
        all_targets_tensor = torch.from_numpy(all_targets)
        
        # Inverse transform - now returns torch tensors
        all_preds = normalizer.inverse_transform_targets(all_preds_tensor).numpy()
        all_targets = normalizer.inverse_transform_targets(all_targets_tensor).numpy()
    
    # Save predictions if requested
    if save_predictions and save_path:
        np.savetxt(save_path, all_preds, delimiter='\t')
    
    # Calculate metrics on the transformed predictions
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(val_loader.dataset)
    
    return metrics

def calculate_metrics(targets, predictions):
    return {
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions),
        'evs': explained_variance_score(targets, predictions)
    }

def get_lr_schedule(optimizer, config):
    def lr_lambda(step):
        if step < config.warmup_epochs:
            return float(step) / float(max(1, config.warmup_epochs))
        
        # Cyclic decay after warmup
        step = step - config.warmup_epochs
        cycle = 1 + step // (config.num_epochs // 4)
        x = step % (config.num_epochs // 4)
        decay = config.cycle_decay ** (cycle - 1)
        
        return max(
            config.min_lr / config.base_lr,
            0.5 * (1 + math.cos(math.pi * x / (config.num_epochs // 4))) * decay
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_data():
    """Load data with ESM-2 feature dimensions"""
    # Load training data
    with open(DATA_PATHS['train_data']) as f:
        reader = csv.reader(f, delimiter='\t')
        datalist_train = [row for row in reader]
    del(datalist_train[0])  # remove header

    # Load test data
    with open(DATA_PATHS['test_data']) as f:
        reader = csv.reader(f, delimiter='\t')
        datalist_test = [row for row in reader]
    del(datalist_test[0])   # remove header

    # Load sequence representations
    sequence_representations_train = torch.load(DATA_PATHS['train_sequence'])
    sequence_representations_test = torch.load(DATA_PATHS['test_sequence'])

    # Print feature dimensions for debugging
    print(f"Training sequence features shape: {len(sequence_representations_train)} sequences")
    print(f"First training sequence feature dimension: {sequence_representations_train[0].size(0)}")
    print(f"Test sequence features shape: {len(sequence_representations_test)} sequences")
    print(f"First test sequence feature dimension: {sequence_representations_test[0].size(0)}")

    # Create normalizer
    normalizer = FeatureNormalizer()
    
    # Prepare training data - convert to tensors once
    train_seq = torch.stack(sequence_representations_train)
    train_z = torch.tensor([float(row[3]) for row in datalist_train], dtype=torch.float32)
    train_mass = torch.tensor([float(row[4]) for row in datalist_train], dtype=torch.float32)
    train_cm = torch.stack([train_z, train_mass], dim=1)
    train_ccs = torch.tensor([float(row[2]) for row in datalist_train], dtype=torch.float32)
    
    # Fit and transform training data
    normalizer.fit(train_seq, train_cm, train_ccs)
    train_seq_norm, train_cm_norm, train_ccs_norm = normalizer.transform(train_seq, train_cm, train_ccs)
    
    # Transform test data - convert to tensors once
    test_seq = torch.stack(sequence_representations_test)
    test_z = torch.tensor([float(row[3]) for row in datalist_test], dtype=torch.float32)
    test_mass = torch.tensor([float(row[4]) for row in datalist_test], dtype=torch.float32)
    test_cm = torch.stack([test_z, test_mass], dim=1)
    test_ccs = torch.tensor([float(row[2]) for row in datalist_test], dtype=torch.float32)
    
    test_seq_norm, test_cm_norm, test_ccs_norm = normalizer.transform(test_seq, test_cm, test_ccs)
    
    # Create datasets
    dataset_train = TensorDataset(train_cm_norm, train_seq_norm, train_ccs_norm)
    dataset_test = TensorDataset(test_cm_norm, test_seq_norm, test_ccs_norm)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(dataset_train)}")
    print(f"Test samples: {len(dataset_test)}")
    
    return train_loader, test_loader, normalizer

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data with normalization
    train_loader, test_loader, normalizer = load_data()
    
    # Get feature dimension from the data
    sample_seq = next(iter(train_loader))[1]  # Get sequence features from first batch
    esm_dim = sample_seq.size(-1)
    print(f"Detected ESM feature dimension: {esm_dim}")
    
    # Create model with correct feature dimension
    model = create_model(config, device, esm_dim)
    
    # Initialize optimizer and criterion
    if config.model_type == 'both':
        optimizer = {
            'improved': torch.optim.AdamW(
                model['improved'].parameters(),
                lr=config.base_lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999)
            ),
            'ensemble': torch.optim.AdamW(
                model['ensemble'].parameters(),
                lr=config.base_lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999)
            )
        }
        scheduler = {
            'improved': get_lr_schedule(optimizer['improved'], config),
            'ensemble': get_lr_schedule(optimizer['ensemble'], config)
        }
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        scheduler = get_lr_schedule(optimizer, config)
    
    criterion = HuberMSELoss()
    
    print("Model created successfully!")
    print(f"Model type: {config.model_type}")
    print(f"Feature dimension: {esm_dim}")

if __name__ == '__main__':
    main()


