"""
@author: a-nakai-k (original code)
Enhanced version with improved architectures, error logging, and visualization

Code for CCS value prediction using preprocessed sequences with improved model architecture.
Features:
- Multiple model architectures (improved, ensemble, combined)
- Comprehensive error logging and metrics tracking
- Training visualization with smoothed curves
- Advanced training techniques and cross-validation
- Extensive visualization and analysis tools
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

# Data paths
DATA_PATHS = {
    'train_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1.tsv',
    'test_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1.tsv',
    'train_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/sequenceTensor_mhcI_train_a1000b1gamma0.pt',
    'test_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/sequenceTensor_mhcI_test_a1000b1gamma0.pt',
    'results': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results'
}

class TrainingConfig:
    def __init__(self):
        # Basic parameters
        self.bs = 200
        self.base_lr = 0.0001
        self.num_epochs = 400
        self.warmup_epochs = 10
        self.patience = 15
        self.accumulation_steps = 8
        self.ema_decay = 0.999
        self.max_grad_norm = 0.5
        
        # Model selection
        self.model_type = 'both'  # Options: 'improved', 'ensemble', 'both'
        self.ensemble_size = 3
        self.ensemble_weights = None  # Will be set based on validation performance
        
        # Advanced parameters
        self.weight_decay = 0.1
        self.label_smoothing = 0.1
        self.dropout_rate = 0.2
        self.hidden_dim = 256
        self.num_folds = 5
        
        # Learning rate schedule
        self.min_lr = 1e-6
        self.cycle_momentum = True
        self.cycle_decay = 0.8
        
        # SWA parameters
        self.swa_start = 100
        self.swa_lr = 0.05
        self.swa_anneal_epochs = 10
        
        # Regularization
        self.mixup_alpha = 0.2
        self.gradient_clip_val = 0.5
        
        # Ensemble specific
        self.diversity_weight = 0.1  # Weight for ensemble diversity loss
        self.temperature = 2.0  # Temperature for ensemble soft predictions

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

# Advanced loss function with focal loss and label smoothing
class FocalMSELoss(_Loss):
    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean'):
        super(FocalMSELoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, input, target):
        mse = F.mse_loss(input, target, reduction='none')
        focal_weight = torch.exp(-self.alpha * mse) ** self.gamma
        loss = focal_weight * mse
        return loss.mean()

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

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

# Improved model architecture
class ImprovedCCSPredictor(nn.Module):
    def __init__(self, config, esm_dim=1280*2):
        super(ImprovedCCSPredictor, self).__init__()
        
        self.dropout_rate = config.dropout_rate
        hidden_dim = config.hidden_dim
        
        # Sequence processor with enhanced residual connections
        self.sequence_processor = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            EnhancedResidualBlock(hidden_dim, hidden_dim * 2, self.dropout_rate),
            EnhancedResidualBlock(hidden_dim, hidden_dim * 2, self.dropout_rate),
            nn.LayerNorm(hidden_dim)
        )
        
        # Enhanced charge/mass processor
        self.charge_mass_processor = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate/2),
            EnhancedResidualBlock(64, 128, self.dropout_rate/2),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # Advanced predictor with skip connections
        self.predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate/2)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU()
            )
        ])
        
        # Final prediction layer to output a single value
        self.final_layer = nn.Linear(hidden_dim // 4, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, seq_features, charge_mass):
        # Apply advanced regularization during training
        if self.training:
            seq_features = F.dropout2d(seq_features.unsqueeze(-1), p=0.1, training=True).squeeze(-1)
            charge_mass = F.dropout(charge_mass, p=0.05, training=True)
        
        # Process features
        seq_processed = self.sequence_processor(seq_features)
        cm_processed = self.charge_mass_processor(charge_mass)
        
        # Forward with skip connections
        x = torch.cat([seq_processed, cm_processed], dim=1)
        intermediate = []
        for layer in self.predictor:
            x = layer(x)
            intermediate.append(x)
        
        # Final prediction
        x = self.final_layer(x)
        
        # Add skip connection from first to last layer if dimensions match
        if len(intermediate) > 1:
            skip_connection = F.adaptive_avg_pool1d(intermediate[0].unsqueeze(-1), 1).squeeze(-1)
            if skip_connection.shape[1] == x.shape[1]:  # Only add if dimensions match
                x = x + 0.1 * skip_connection
        
        return x

class EnhancedEnsembleCCSPredictor(nn.Module):
    def __init__(self, config, esm_dim=1280*2):
        super(EnhancedEnsembleCCSPredictor, self).__init__()
        self.models = nn.ModuleList([
            ImprovedCCSPredictor(config, esm_dim) 
            for _ in range(config.ensemble_size)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(config.ensemble_size) / config.ensemble_size
        )
        self.temperature = config.temperature
        
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
        predictions = torch.cat([p.unsqueeze(1) for p in predictions], dim=1)
        mean_pred = torch.mean(predictions, dim=1, keepdim=True)
        diversity_loss = -torch.mean(torch.pow(predictions - mean_pred, 2))
        return diversity_loss

def create_model(config, device):
    if config.model_type == 'improved':
        return ImprovedCCSPredictor(config).to(device)
    elif config.model_type == 'ensemble':
        return EnhancedEnsembleCCSPredictor(config).to(device)
    elif config.model_type == 'both':
        return {
            'improved': ImprovedCCSPredictor(config).to(device),
            'ensemble': EnhancedEnsembleCCSPredictor(config).to(device)
        }
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler, epoch):
    if isinstance(model, dict):
        return train_epoch_both(model, train_loader, optimizer, criterion, device, config, scheduler, epoch)
    
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    optimizer.zero_grad()
    
    for batch_idx, (charge_mass, seq, ccs) in enumerate(train_loader):
        charge_mass = charge_mass.to(device, dtype=torch.float)
        seq = seq.to(device)
        ccs = ccs.to(device, dtype=torch.float)
        
        # Apply mixup augmentation
        if config.mixup_alpha > 0 and epoch >= config.warmup_epochs:
            seq, ccs = mixup_data(seq, ccs, config.mixup_alpha)
        
        # Forward pass
        if isinstance(model, EnhancedEnsembleCCSPredictor):
            output, ensemble_preds = model(seq, charge_mass)
            loss = criterion(output, ccs.view(-1, 1))
            # Add diversity loss for ensemble
            diversity_loss = model.get_diversity_loss(ensemble_preds)
            loss = loss + config.diversity_weight * diversity_loss
        else:
            output = model(seq, charge_mass)
            loss = criterion(output, ccs.view(-1, 1))
        
        # Scale loss for gradient accumulation
        loss = loss / config.accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_val
            )
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * len(charge_mass) * config.accumulation_steps
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(ccs.view(-1, 1).detach().cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(train_loader.dataset)
    
    return metrics

def train_epoch_both(models, train_loader, optimizers, criterion, device, config, schedulers, epoch):
    metrics_improved = train_epoch(
        models['improved'], 
        train_loader, 
        optimizers['improved'], 
        criterion, 
        device, 
        config, 
        schedulers['improved'], 
        epoch
    )
    
    metrics_ensemble = train_epoch(
        models['ensemble'], 
        train_loader, 
        optimizers['ensemble'], 
        criterion, 
        device, 
        config, 
        schedulers['ensemble'], 
        epoch
    )
    
    return {
        'improved': metrics_improved,
        'ensemble': metrics_ensemble
    }

def validate(model, val_loader, criterion, device):
    if isinstance(model, dict):
        return validate_both(model, val_loader, criterion, device)
        
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for charge_mass, seq, ccs in val_loader:
            charge_mass = charge_mass.to(device, dtype=torch.float)
            seq = seq.to(device)
            ccs = ccs.to(device, dtype=torch.float)
            
            if isinstance(model, EnhancedEnsembleCCSPredictor):
                output = model(seq, charge_mass)  # No need for ensemble_preds in eval
            else:
                output = model(seq, charge_mass)
                
            loss = criterion(output, ccs.view(-1, 1))
            
            total_loss += loss.item() * len(charge_mass)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(ccs.view(-1, 1).cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(val_loader.dataset)
    
    return metrics

def validate_both(models, val_loader, criterion, device):
    metrics_improved = validate(models['improved'], val_loader, criterion, device)
    metrics_ensemble = validate(models['ensemble'], val_loader, criterion, device)
    
    # Combine predictions using learned weights
    if models['ensemble'].ensemble_weights is not None:
        combined_metrics = validate_combined(
            models['improved'],
            models['ensemble'],
            val_loader,
            criterion,
            device
        )
        return {
            'improved': metrics_improved,
            'ensemble': metrics_ensemble,
            'combined': combined_metrics
        }
    
    return {
        'improved': metrics_improved,
        'ensemble': metrics_ensemble
    }

def validate_combined(improved_model, ensemble_model, val_loader, criterion, device):
    improved_model.eval()
    ensemble_model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for charge_mass, seq, ccs in val_loader:
            charge_mass = charge_mass.to(device, dtype=torch.float)
            seq = seq.to(device)
            ccs = ccs.to(device, dtype=torch.float)
            
            # Get predictions from both models
            improved_pred = improved_model(seq, charge_mass)
            ensemble_pred = ensemble_model(seq, charge_mass)
            
            # Combine predictions
            combined_pred = 0.5 * (improved_pred + ensemble_pred)
            loss = criterion(combined_pred, ccs.view(-1, 1))
            
            total_loss += loss.item() * len(charge_mass)
            all_preds.extend(combined_pred.cpu().numpy())
            all_targets.extend(ccs.view(-1, 1).cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(val_loader.dataset)
    
    return metrics

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

def calculate_metrics(targets, predictions):
    return {
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions),
        'evs': explained_variance_score(targets, predictions)
    }

def train_with_cv(config, train_loader, test_loader, device, normalizer):
    kfold = KFold(n_splits=config.num_folds, shuffle=True)
    fold_results = []
    
    train_dataset = train_loader.dataset
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold+1}/{config.num_folds}')
        print('-----------------------------------')
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        fold_train_loader = DataLoader(
            train_dataset, 
            batch_size=config.bs,
            sampler=train_sampler
        )
        fold_val_loader = DataLoader(
            train_dataset,
            batch_size=config.bs,
            sampler=val_sampler
        )
        
        # Initialize model(s) and training components
        model = create_model(config, device)
        criterion = FocalMSELoss()
        
        if isinstance(model, dict):
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
            
            swa_model = {
                'improved': AveragedModel(model['improved']),
                'ensemble': AveragedModel(model['ensemble'])
            }
            
            swa_scheduler = {
                'improved': SWALR(optimizer['improved'], swa_lr=config.swa_lr),
                'ensemble': SWALR(optimizer['ensemble'], swa_lr=config.swa_lr)
            }
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.base_lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999)
            )
            scheduler = get_lr_schedule(optimizer, config)
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        fold_history = initialize_fold_history(config.model_type)
        
        for epoch in range(config.num_epochs):
            # Training
            train_metrics = train_epoch(
                model, fold_train_loader, optimizer, criterion,
                device, config, scheduler, epoch
            )
            
            # Validation
            val_metrics = validate(model, fold_val_loader, criterion, device)
            
            # Update SWA model
            if epoch >= config.swa_start:
                if isinstance(model, dict):
                    for model_type in model:
                        swa_model[model_type].update_parameters(model[model_type])
                        swa_scheduler[model_type].step()
                else:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
            else:
                if isinstance(scheduler, dict):
                    for sched in scheduler.values():
                        sched.step()
                else:
                    scheduler.step()
            
            # Record metrics
            update_fold_history(fold_history, train_metrics, val_metrics, config.model_type)
            
            # Early stopping check
            current_val_rmse = get_validation_rmse(val_metrics, config.model_type)
            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
            
            print_epoch_metrics(epoch, train_metrics, val_metrics, config.model_type)
        
        # Final evaluation with SWA model
        if isinstance(model, dict):
            swa_results = {
                model_type: validate(swa_model[model_type], test_loader, criterion, device)
                for model_type in model
            }
        else:
            swa_results = validate(swa_model, test_loader, criterion, device)
            
        fold_results.append({
            'fold': fold + 1,
            'best_val_rmse': best_val_rmse,
            'swa_results': swa_results,
            'history': fold_history
        })
    
    return fold_results

def initialize_fold_history(model_type):
    if model_type == 'both':
        return {
            'improved': {metric: [] for metric in ['train_rmse', 'val_rmse', 'train_r2', 'val_r2']},
            'ensemble': {metric: [] for metric in ['train_rmse', 'val_rmse', 'train_r2', 'val_r2']},
            'combined': {metric: [] for metric in ['val_rmse', 'val_r2']}
        }
    return {metric: [] for metric in ['train_rmse', 'val_rmse', 'train_r2', 'val_r2']}

def update_fold_history(history, train_metrics, val_metrics, model_type):
    if model_type == 'both':
        for model_type in ['improved', 'ensemble']:
            for metric in history[model_type]:
                if 'train' in metric:
                    history[model_type][metric].append(
                        train_metrics[model_type][metric.replace('train_', '')]
                    )
                else:
                    history[model_type][metric].append(
                        val_metrics[model_type][metric.replace('val_', '')]
                    )
        if 'combined' in val_metrics:
            for metric in history['combined']:
                history['combined'][metric].append(val_metrics['combined'][metric.replace('val_', '')])
    else:
        for metric in history:
            if 'train' in metric:
                history[metric].append(train_metrics[metric.replace('train_', '')])
            else:
                history[metric].append(val_metrics[metric.replace('val_', '')])

def get_validation_rmse(val_metrics, model_type):
    if model_type == 'both':
        # Use the better of the two models or their combination
        rmse_improved = val_metrics['improved']['rmse']
        rmse_ensemble = val_metrics['ensemble']['rmse']
        if 'combined' in val_metrics:
            rmse_combined = val_metrics['combined']['rmse']
            return min(rmse_improved, rmse_ensemble, rmse_combined)
        return min(rmse_improved, rmse_ensemble)
    return val_metrics['rmse']

def print_epoch_metrics(epoch, train_metrics, val_metrics, model_type):
    if model_type == 'both':
        print(f"\nEpoch {epoch+1}:")
        print("Improved Model:")
        print(f"Train RMSE = {train_metrics['improved']['rmse']:.4f}, "
              f"Val RMSE = {val_metrics['improved']['rmse']:.4f}")
        print(f"Train R² = {train_metrics['improved']['r2']:.4f}, "
              f"Val R² = {val_metrics['improved']['r2']:.4f}")
        
        print("\nEnsemble Model:")
        print(f"Train RMSE = {train_metrics['ensemble']['rmse']:.4f}, "
              f"Val RMSE = {val_metrics['ensemble']['rmse']:.4f}")
        print(f"Train R² = {train_metrics['ensemble']['r2']:.4f}, "
              f"Val R² = {val_metrics['ensemble']['r2']:.4f}")
        
        if 'combined' in val_metrics:
            print("\nCombined Model:")
            print(f"Val RMSE = {val_metrics['combined']['rmse']:.4f}, "
                  f"Val R² = {val_metrics['combined']['r2']:.4f}")
    else:
        print(f"Epoch {epoch+1}: "
              f"Train RMSE = {train_metrics['rmse']:.4f}, "
              f"Val RMSE = {val_metrics['rmse']:.4f}, "
              f"Train R² = {train_metrics['r2']:.4f}, "
              f"Val R² = {val_metrics['r2']:.4f}")

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Load and prepare data with normalization
    train_loader, test_loader, normalizer = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Perform cross-validation training
    cv_results = train_with_cv(config, train_loader, test_loader, device, normalizer)
    
    # Analyze and save results
    analyze_cv_results(cv_results, config)

def analyze_cv_results(cv_results, config):
    """Enhanced analysis function with additional visualizations."""
    # Calculate average metrics across folds
    avg_metrics = {
        'test_rmse': np.mean([r['test_rmse'] for r in cv_results]),
        'test_r2': np.mean([r['test_r2'] for r in cv_results]),
        'best_val_rmse': np.mean([r['best_val_rmse'] for r in cv_results])
    }
    
    # Calculate standard deviations
    std_metrics = {
        'test_rmse': np.std([r['test_rmse'] for r in cv_results]),
        'test_r2': np.std([r['test_r2'] for r in cv_results]),
        'best_val_rmse': np.std([r['best_val_rmse'] for r in cv_results])
    }
    
    # Print summary
    print("\nCross-Validation Results Summary:")
    print(f"Average Test RMSE: {avg_metrics['test_rmse']:.4f} ± {std_metrics['test_rmse']:.4f}")
    print(f"Average Test R²: {avg_metrics['test_r2']:.4f} ± {std_metrics['test_r2']:.4f}")
    print(f"Average Best Validation RMSE: {avg_metrics['best_val_rmse']:.4f} ± {std_metrics['best_val_rmse']:.4f}")
    
    # Plot learning curves for each fold
    plot_cv_learning_curves(cv_results, config)
    
    # Create prediction analysis plots for each model type
    plot_dir = "model_analysis_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    if config.model_type == 'both':
        for model_type in ['improved', 'ensemble', 'combined']:
            if os.path.exists(f'out_test_predictCCS_mhcI_{model_type}.csv'):
                plot_prediction_analysis(
                    'test_1.tsv',
                    f'out_test_predictCCS_mhcI_{model_type}.csv',
                    plot_dir,
                    model_type
                )
    else:
        plot_prediction_analysis(
            'test_1.tsv',
            f'out_test_predictCCS_mhcI_{config.model_type}.csv',
            plot_dir,
            config.model_type
        )

def plot_cv_learning_curves(cv_results, config):
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE
    plt.subplot(2, 1, 1)
    for fold_idx, result in enumerate(cv_results):
        plt.plot(result['history']['train_rmse'], 
                label=f'Fold {fold_idx+1} Train',
                alpha=0.3)
        plt.plot(result['history']['val_rmse'],
                label=f'Fold {fold_idx+1} Val',
                alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Learning Curves Across Folds')
    plt.legend()
    
    # Plot R²
    plt.subplot(2, 1, 2)
    for fold_idx, result in enumerate(cv_results):
        plt.plot(result['history']['train_r2'],
                label=f'Fold {fold_idx+1} Train',
                alpha=0.3)
        plt.plot(result['history']['val_r2'],
                label=f'Fold {fold_idx+1} Val',
                alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Learning Curves Across Folds')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cv_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_analysis(experimental_path, predicted_path, plot_dir, model_type):
    """
    Create comprehensive prediction analysis plots including scatter plots,
    residual analysis, and calibration curves.
    """
    # Load datasets
    predicted_df = pd.read_csv(predicted_path, sep='\t', header=None)
    predicted_df.columns = ["CCS_Predicted"]
    
    experimental_df = pd.read_csv(experimental_path, sep="\t")
    experimental_df = experimental_df[["CCS_Experimental"]]
    
    # Convert to float
    predicted_df['CCS_Predicted'] = predicted_df['CCS_Predicted'].astype(float)
    experimental_df['CCS_Experimental'] = experimental_df['CCS_Experimental'].astype(float)
    
    # Ensure same length
    if len(predicted_df) != len(experimental_df):
        raise ValueError(f"Datasets have different lengths. Predicted: {len(predicted_df)}, Experimental: {len(experimental_df)}.")
    
    # Combine into one dataframe
    df = pd.concat([experimental_df, predicted_df], axis=1)
    
    # Create all plots
    plot_scatter_with_regression(df, plot_dir, model_type)
    plot_residuals(df, plot_dir, model_type)
    plot_calibration_curve(df, plot_dir, model_type)
    plot_residual_distribution(df, plot_dir, model_type)

def plot_scatter_with_regression(df, plot_dir, model_type):
    """Create scatter plot with regression line and error bands."""
    X = df["CCS_Experimental"].values.reshape(-1, 1)
    y = df["CCS_Predicted"].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Create smooth range for regression line
    x_range = np.linspace(df["CCS_Experimental"].min(), df["CCS_Experimental"].max(), 500).reshape(-1, 1)
    y_pred_line = model.predict(x_range)
    
    # Compute metrics
    r2 = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    
    plt.figure(figsize=(8, 6))
    
    # Scatter points
    plt.scatter(df["CCS_Experimental"], df["CCS_Predicted"], color='black', s=10, label='Data')
    
    # Regression line
    plt.plot(x_range, y_pred_line, color='red', label='Regression line')
    
    # Identity line
    min_val = min(df["CCS_Experimental"].min(), df["CCS_Predicted"].min())
    max_val = max(df["CCS_Experimental"].max(), df["CCS_Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='Ideal prediction (y = x)')
    
    # Error bands
    plt.fill_between(x_range.flatten(), y_pred_line - rmse, y_pred_line + rmse,
                     color='gray', alpha=0.2, label='±1 RMSE')
    
    plt.xlabel("Experimental CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"Predicted vs Experimental CCS ({model_type})")
    plt.text(0.95, 0.05, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
             ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black'))
    
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'scatter_regression_{model_type}.png'), dpi=400)
    plt.close()
    
    return r2, mae, rmse

def plot_residuals(df, plot_dir, model_type):
    """Create residual plot."""
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['CCS_Experimental'], y=df['Residuals'], color='black')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Experimental CCS')
    plt.ylabel('Residuals (Predicted - Experimental)')
    plt.title(f'Residual Plot: {model_type}')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'residuals_{model_type}.png'), dpi=400)
    plt.close()

def plot_calibration_curve(df, plot_dir, model_type):
    """Create calibration curve."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['CCS_Experimental'], y=df['CCS_Predicted'], color='black')
    plt.plot([df['CCS_Experimental'].min(), df['CCS_Experimental'].max()],
             [df['CCS_Experimental'].min(), df['CCS_Experimental'].max()],
             color='red', linestyle='--')
    
    r2 = r2_score(df['CCS_Experimental'], df['CCS_Predicted'])
    mae = mean_absolute_error(df['CCS_Experimental'], df['CCS_Predicted'])
    
    plt.xlabel('Experimental CCS')
    plt.ylabel('Predicted CCS')
    plt.title(f'Calibration Curve: {model_type}')
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
             ha='left', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black'))
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'calibration_{model_type}.png'), dpi=400)
    plt.close()

def plot_residual_distribution(df, plot_dir, model_type):
    """Create residual distribution plot."""
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Residuals'], bins=30, kde=True, color='gray')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residuals (Predicted - Experimental)')
    plt.title(f'Distribution of Residuals: {model_type}')
    
    # Add mean and std annotations
    mean_residual = df['Residuals'].mean()
    std_residual = df['Residuals'].std()
    plt.text(0.95, 0.95, f"Mean = {mean_residual:.3f}\nStd = {std_residual:.3f}",
             ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black'))
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'residual_distribution_{model_type}.png'), dpi=400)
    plt.close()

def load_data():
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

    # Create normalizer
    normalizer = FeatureNormalizer()
    
    # Prepare training data
    train_seq = torch.stack(sequence_representations_train)
    
    # Convert string data to float tensors
    train_z = torch.tensor([float(row[3]) for row in datalist_train], dtype=torch.float32)
    train_mass = torch.tensor([float(row[4]) for row in datalist_train], dtype=torch.float32)
    train_cm = torch.stack([train_z, train_mass], dim=1)
    train_ccs = torch.tensor([float(row[2]) for row in datalist_train], dtype=torch.float32)
    
    # Fit and transform training data
    normalizer.fit(train_seq, train_cm, train_ccs)
    train_seq_norm, train_cm_norm, train_ccs_norm = normalizer.transform(train_seq, train_cm, train_ccs)
    
    # Transform test data
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
    train_loader = DataLoader(dataset_train, batch_size=200, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=200, shuffle=False)
    
    return train_loader, test_loader, normalizer

if __name__ == '__main__':
    main() 
