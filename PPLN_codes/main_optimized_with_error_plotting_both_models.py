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

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input data and target."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    # No need for explicit device placement - TPU will handle it
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

def create_model(config, device):
    if config.model_type == 'improved':
        return ImprovedCCSPredictor(config).to(device)
    elif config.model_type == 'ensemble':
        return EnhancedEnsembleCCSPredictor(config).to(device)
    elif config.model_type == 'both':
        improved_model = ImprovedCCSPredictor(config).to(device)
        ensemble_model = EnhancedEnsembleCCSPredictor(config).to(device)
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
    
    import torch_xla.core.xla_model as xm
    
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
            xm.optimizer_step(optimizer)
            xm.mark_step()
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

def validate(model, val_loader, criterion, device, save_predictions=False, save_path=None, normalizer=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    import torch_xla.core.xla_model as xm
    
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

def validate_both(models, val_loader, criterion, device, normalizer=None, save_predictions=False, save_path=None):
    metrics_improved = validate(models['improved'], val_loader, criterion, device, normalizer=normalizer)
    metrics_ensemble = validate(models['ensemble'], val_loader, criterion, device, normalizer=normalizer)
    
    # Combine predictions using learned weights
    if models['ensemble'].ensemble_weights is not None:
        combined_metrics = validate_combined(
            models['improved'],
            models['ensemble'],
            val_loader,
            criterion,
            device,
            normalizer=normalizer,
            save_predictions=save_predictions,
            save_path=save_path
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

def validate_combined(improved_model, ensemble_model, val_loader, criterion, device, normalizer=None, save_predictions=False, save_path=None):
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
            
            # Combine predictions using config weights
            combined_pred = (improved_pred * improved_model.config.combined_weights['improved'] + 
                           ensemble_pred * improved_model.config.combined_weights['ensemble'])
            
            loss = criterion(combined_pred, ccs.view(-1, 1))
            
            total_loss += loss.item() * len(charge_mass)
            all_preds.extend(combined_pred.cpu().numpy())
            all_targets.extend(ccs.view(-1, 1).cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Inverse transform if normalizer provided
    if normalizer is not None:
        all_preds_tensor = torch.from_numpy(all_preds)
        all_targets_tensor = torch.from_numpy(all_targets)
        all_preds = normalizer.inverse_transform_targets(all_preds_tensor).numpy()
        all_targets = normalizer.inverse_transform_targets(all_targets_tensor).numpy()
    
    # Save predictions if requested
    if save_predictions and save_path:
        # Create directory for predictions if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, all_preds, delimiter='\t')
    
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
    # Create results directory if it doesn't exist
    results_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different result types
    plots_dir = os.path.join(results_dir, f"plots_{config.model_type}")
    predictions_dir = os.path.join(results_dir, f"predictions_{config.model_type}")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
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
            sampler=train_sampler,
            num_workers=0
        )
        fold_val_loader = DataLoader(
            train_dataset,
            batch_size=config.bs,
            sampler=val_sampler,
            num_workers=0
        )
        
        # Initialize model and move to TPU
        model = create_model(config, device)
        
        # Initialize optimizer and criterion differently for 'both' model type
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
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        fold_history = initialize_fold_history(config.model_type)
        best_model_state = None if config.model_type != 'both' else {'improved': None, 'ensemble': None}
        
        for epoch in range(config.num_epochs):
            # Training
            if config.model_type == 'both':
                train_metrics = train_epoch(model, fold_train_loader, optimizer, criterion, device, config, scheduler, epoch)
                val_metrics = validate_both(model, fold_val_loader, criterion, device, normalizer=normalizer)
                current_val_rmse = min(
                    val_metrics['improved']['rmse'],
                    val_metrics['ensemble']['rmse'],
                    val_metrics.get('combined', {'rmse': float('inf')})['rmse']
                )
            else:
                train_metrics = train_epoch(model, fold_train_loader, optimizer, criterion, device, config, scheduler, epoch)
                val_metrics = validate(model, fold_val_loader, criterion, device, normalizer=normalizer)
                current_val_rmse = val_metrics['rmse']
            
            # For TPU, we need to wait for computations to finish
            if hasattr(device, 'type') and device.type == 'xla':
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            
            # Step the scheduler
            if config.model_type == 'both':
                for s in scheduler.values():
                    s.step()
            else:
                scheduler.step()
            
            # Record metrics
            update_fold_history(fold_history, train_metrics, val_metrics, config.model_type)
            
            # Early stopping check with debug info
            if current_val_rmse < best_val_rmse:
                improvement = best_val_rmse - current_val_rmse
                best_val_rmse = current_val_rmse
                print(f"\nRMSE improved by {improvement:.6f}")
                patience_counter = 0
                # Save best model state
                if config.model_type == 'both':
                    best_model_state = {
                        'improved': {k: v.cpu() for k, v in model['improved'].state_dict().items()},
                        'ensemble': {k: v.cpu() for k, v in model['ensemble'].state_dict().items()}
                    }
                    # Save both models
                    for model_key in ['improved', 'ensemble']:
                        torch.save(
                            best_model_state[model_key],
                            os.path.join(results_dir, f'best_model_{model_key}_fold_{fold+1}.pt')
                        )
                else:
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(best_model_state, os.path.join(results_dir, f'best_model_fold_{fold+1}_{config.model_type}.pt'))
            else:
                patience_counter += 1
                print(f"\nNo improvement. Patience: {patience_counter}/{config.patience}")
                
            if patience_counter >= config.patience:
                print(f'\nEarly stopping triggered! No improvement for {config.patience} epochs.')
                print(f'Best validation RMSE: {best_val_rmse:.4f}')
                break
            
            print_epoch_metrics(epoch, train_metrics, val_metrics, config.model_type)
        
        # Load best model for final evaluation
        if best_model_state is not None:
            if config.model_type == 'both':
                model['improved'].load_state_dict(best_model_state['improved'])
                model['ensemble'].load_state_dict(best_model_state['ensemble'])
            else:
                model.load_state_dict(best_model_state)
        
        # Final evaluation and save predictions
        predictions_path = os.path.join(predictions_dir, f'predictions_fold_{fold+1}.tsv')
        if config.model_type == 'both':
            test_metrics = validate_both(model, test_loader, criterion, device, 
                                       save_predictions=True, save_path=predictions_path,
                                       normalizer=normalizer)
        else:
            test_metrics = validate(model, test_loader, criterion, device, 
                                  save_predictions=True, save_path=predictions_path,
                                  normalizer=normalizer)
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_rmse': best_val_rmse,
            'test_metrics': test_metrics,
            'history': fold_history,
            'predictions_path': predictions_path
        })
        
        # Save fold history
        if config.model_type == 'both':
            # Save separate history files for each model type
            for model_type in ['improved', 'ensemble']:
                history_df = pd.DataFrame({
                    'epoch': range(len(fold_history[model_type]['train_rmse'])),
                    'train_rmse': fold_history[model_type]['train_rmse'],
                    'val_rmse': fold_history[model_type]['val_rmse'],
                    'train_r2': fold_history[model_type]['train_r2'],
                    'val_r2': fold_history[model_type]['val_r2']
                })
                history_df.to_csv(
                    os.path.join(results_dir, f'history_fold_{fold+1}_{model_type}.csv'),
                    index=False
                )
            
            # Save combined model history if available
            if 'combined' in fold_history:
                history_df = pd.DataFrame({
                    'epoch': range(len(fold_history['combined']['val_rmse'])),
                    'val_rmse': fold_history['combined']['val_rmse'],
                    'val_r2': fold_history['combined']['val_r2']
                })
                history_df.to_csv(
                    os.path.join(results_dir, f'history_fold_{fold+1}_combined.csv'),
                    index=False
                )
        else:
            history_df = pd.DataFrame({
                'epoch': range(len(fold_history['train_rmse'])),
                'train_rmse': fold_history['train_rmse'],
                'val_rmse': fold_history['val_rmse'],
                'train_r2': fold_history['train_r2'],
                'val_r2': fold_history['val_r2']
            })
            history_df.to_csv(
                os.path.join(results_dir, f'history_fold_{fold+1}_{config.model_type}.csv'),
                index=False
            )
    
    return fold_results

def initialize_fold_history(model_type):
    """Initialize the history dictionary based on model type."""
    if model_type == 'both':
        return {
            'improved': {
                'train_rmse': [],
                'val_rmse': [],
                'train_r2': [],
                'val_r2': [],
                'train_loss': [],
                'val_loss': []
            },
            'ensemble': {
                'train_rmse': [],
                'val_rmse': [],
                'train_r2': [],
                'val_r2': [],
                'train_loss': [],
                'val_loss': []
            },
            'combined': {
                'val_rmse': [],
                'val_r2': [],
                'val_loss': []
            }
        }
    return {
        'train_rmse': [],
        'val_rmse': [],
        'train_r2': [],
        'val_r2': [],
        'train_loss': [],
        'val_loss': []
    }

def update_fold_history(history, train_metrics, val_metrics, model_type):
    """Update the history dictionary with new metrics."""
    if model_type == 'both':
        # Update metrics for each model type
        for model_key in ['improved', 'ensemble']:
            if model_key in train_metrics and model_key in val_metrics:
                history[model_key]['train_rmse'].append(train_metrics[model_key]['rmse'])
                history[model_key]['val_rmse'].append(val_metrics[model_key]['rmse'])
                history[model_key]['train_r2'].append(train_metrics[model_key]['r2'])
                history[model_key]['val_r2'].append(val_metrics[model_key]['r2'])
                history[model_key]['train_loss'].append(train_metrics[model_key]['loss'])
                history[model_key]['val_loss'].append(val_metrics[model_key]['loss'])
        
        # Update combined metrics if available
        if 'combined' in val_metrics:
            history['combined']['val_rmse'].append(val_metrics['combined']['rmse'])
            history['combined']['val_r2'].append(val_metrics['combined']['r2'])
            history['combined']['val_loss'].append(val_metrics['combined']['loss'])
    else:
        # Update metrics for single model
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_r2'].append(val_metrics['r2'])
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])

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
    """Print metrics for the current epoch."""
    print(f"\nEpoch {epoch+1}:")
    
    if model_type == 'both':
        for model_key in ['improved', 'ensemble']:
            print(f"\n{model_key.capitalize()} Model:")
            print(f"Train RMSE = {train_metrics[model_key]['rmse']:.4f}, "
                  f"Val RMSE = {val_metrics[model_key]['rmse']:.4f}")
            print(f"Train R² = {train_metrics[model_key]['r2']:.4f}, "
                  f"Val R² = {val_metrics[model_key]['r2']:.4f}")
            print(f"Train Loss = {train_metrics[model_key]['loss']:.4f}, "
                  f"Val Loss = {val_metrics[model_key]['loss']:.4f}")
        
        if 'combined' in val_metrics:
            print("\nCombined Model:")
            print(f"Val RMSE = {val_metrics['combined']['rmse']:.4f}, "
                  f"Val R² = {val_metrics['combined']['r2']:.4f}, "
                  f"Val Loss = {val_metrics['combined']['loss']:.4f}")
    else:
        print(f"Train RMSE = {train_metrics['rmse']:.4f}, "
              f"Val RMSE = {val_metrics['rmse']:.4f}")
        print(f"Train R² = {train_metrics['r2']:.4f}, "
              f"Val R² = {val_metrics['r2']:.4f}")
        print(f"Train Loss = {train_metrics['loss']:.4f}, "
              f"Val Loss = {val_metrics['loss']:.4f}")

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Set up TPU if available
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("TPU device detected and initialized")
    except ImportError:
        device = torch.device('cpu')
        print("No TPU found, using CPU")
    
    # Load and prepare data with normalization
    train_loader, test_loader, normalizer = load_data()
    
    # Perform cross-validation training
    cv_results = train_with_cv(config, train_loader, test_loader, device, normalizer)
    
    # Analyze and save results
    analyze_cv_results(cv_results, config)

def analyze_cv_results(cv_results, config):
    """Enhanced analysis function with additional visualizations and proper handling of both model types."""
    # Set up results directory
    results_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results"
    plots_dir = os.path.join(results_dir, f"plots_{config.model_type}")
    os.makedirs(plots_dir, exist_ok=True)
    
    if config.model_type == 'both':
        # Calculate metrics separately for each model type
        avg_metrics = {
            'improved': {
                'test_rmse': np.mean([r['test_metrics']['improved']['rmse'] for r in cv_results]),
                'test_r2': np.mean([r['test_metrics']['improved']['r2'] for r in cv_results]),
                'best_val_rmse': np.mean([r['best_val_rmse'] for r in cv_results])
            },
            'ensemble': {
                'test_rmse': np.mean([r['test_metrics']['ensemble']['rmse'] for r in cv_results]),
                'test_r2': np.mean([r['test_metrics']['ensemble']['r2'] for r in cv_results]),
                'best_val_rmse': np.mean([r['best_val_rmse'] for r in cv_results])
            }
        }
        
        # Calculate standard deviations
        std_metrics = {
            'improved': {
                'test_rmse': np.std([r['test_metrics']['improved']['rmse'] for r in cv_results]),
                'test_r2': np.std([r['test_metrics']['improved']['r2'] for r in cv_results]),
                'best_val_rmse': np.std([r['best_val_rmse'] for r in cv_results])
            },
            'ensemble': {
                'test_rmse': np.std([r['test_metrics']['ensemble']['rmse'] for r in cv_results]),
                'test_r2': np.std([r['test_metrics']['ensemble']['r2'] for r in cv_results]),
                'best_val_rmse': np.std([r['best_val_rmse'] for r in cv_results])
            }
        }
        
        # Add combined metrics if available
        if any('combined' in r['test_metrics'] for r in cv_results):
            avg_metrics['combined'] = {
                'test_rmse': np.mean([r['test_metrics']['combined']['rmse'] for r in cv_results]),
                'test_r2': np.mean([r['test_metrics']['combined']['r2'] for r in cv_results])
            }
            std_metrics['combined'] = {
                'test_rmse': np.std([r['test_metrics']['combined']['rmse'] for r in cv_results]),
                'test_r2': np.std([r['test_metrics']['combined']['r2'] for r in cv_results])
            }
        
        # Save summary metrics for each model type
        for model_type in avg_metrics.keys():
            summary_df = pd.DataFrame({
                'Metric': ['Test RMSE', 'Test R²', 'Best Val RMSE'] if model_type != 'combined' else ['Test RMSE', 'Test R²'],
                'Mean': [avg_metrics[model_type]['test_rmse'], 
                        avg_metrics[model_type]['test_r2'],
                        avg_metrics[model_type].get('best_val_rmse', np.nan)],
                'Std': [std_metrics[model_type]['test_rmse'],
                       std_metrics[model_type]['test_r2'],
                       std_metrics[model_type].get('best_val_rmse', np.nan)]
            })
            summary_df.to_csv(os.path.join(results_dir, f'summary_metrics_{model_type}.csv'), index=False)
        
        # Print summary for each model type
        print("\nCross-Validation Results Summary:")
        for model_type in avg_metrics.keys():
            print(f"\n{model_type.capitalize()} Model:")
            print(f"Average Test RMSE: {avg_metrics[model_type]['test_rmse']:.4f} ± {std_metrics[model_type]['test_rmse']:.4f}")
            print(f"Average Test R²: {avg_metrics[model_type]['test_r2']:.4f} ± {std_metrics[model_type]['test_r2']:.4f}")
            if 'best_val_rmse' in avg_metrics[model_type]:
                print(f"Average Best Validation RMSE: {avg_metrics[model_type]['best_val_rmse']:.4f} ± {std_metrics[model_type]['best_val_rmse']:.4f}")
    
    else:
        # Original handling for single model type
        avg_metrics = {
            'test_rmse': np.mean([r['test_metrics']['rmse'] for r in cv_results]),
            'test_r2': np.mean([r['test_metrics']['r2'] for r in cv_results]),
            'best_val_rmse': np.mean([r['best_val_rmse'] for r in cv_results])
        }
        
        std_metrics = {
            'test_rmse': np.std([r['test_metrics']['rmse'] for r in cv_results]),
            'test_r2': np.std([r['test_metrics']['r2'] for r in cv_results]),
            'best_val_rmse': np.std([r['best_val_rmse'] for r in cv_results])
        }
        
        # Save summary metrics
        summary_df = pd.DataFrame({
            'Metric': ['Test RMSE', 'Test R²', 'Best Val RMSE'],
            'Mean': [avg_metrics['test_rmse'], avg_metrics['test_r2'], avg_metrics['best_val_rmse']],
            'Std': [std_metrics['test_rmse'], std_metrics['test_r2'], std_metrics['best_val_rmse']]
        })
        summary_df.to_csv(os.path.join(results_dir, f'summary_metrics_{config.model_type}.csv'), index=False)
        
        # Print summary
        print("\nCross-Validation Results Summary:")
        print(f"Average Test RMSE: {avg_metrics['test_rmse']:.4f} ± {std_metrics['test_rmse']:.4f}")
        print(f"Average Test R²: {avg_metrics['test_r2']:.4f} ± {std_metrics['test_r2']:.4f}")
        print(f"Average Best Validation RMSE: {avg_metrics['best_val_rmse']:.4f} ± {std_metrics['best_val_rmse']:.4f}")
    
    # Plot learning curves for each fold
    try:
        plot_cv_learning_curves(cv_results, config, plots_dir)
    except Exception as e:
        print(f"Warning: Could not plot learning curves: {str(e)}")
    
    # Plot predictions for each fold
    for result in cv_results:
        try:
            if os.path.exists(result['predictions_path']):
                fold_num = result['fold']
                plot_prediction_analysis(
                    DATA_PATHS['test_data'],
                    result['predictions_path'],
                    plots_dir,
                    f'fold_{fold_num}'
                )
        except Exception as e:
            print(f"Warning: Could not plot predictions for fold {result.get('fold', 'unknown')}: {str(e)}")
            
    print(f"\nAnalysis completed. Results saved in '{results_dir}'")

def plot_cv_learning_curves(cv_results, config, plots_dir):
    """Plot learning curves with proper handling of both model types."""
    if config.model_type == 'both':
        for model_type in ['improved', 'ensemble']:
            plt.figure(figsize=(15, 10))
            
            # Plot RMSE
            plt.subplot(2, 1, 1)
            for fold_idx, result in enumerate(cv_results):
                history = result['history'][model_type]
                plt.plot(history['train_rmse'], 
                        label=f'Fold {fold_idx+1} Train',
                        alpha=0.3)
                plt.plot(history['val_rmse'],
                        label=f'Fold {fold_idx+1} Val',
                        alpha=0.3)
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title(f'RMSE Learning Curves Across Folds ({model_type})')
            plt.legend()
            
            # Plot R²
            plt.subplot(2, 1, 2)
            for fold_idx, result in enumerate(cv_results):
                history = result['history'][model_type]
                plt.plot(history['train_r2'],
                        label=f'Fold {fold_idx+1} Train',
                        alpha=0.3)
                plt.plot(history['val_r2'],
                        label=f'Fold {fold_idx+1} Val',
                        alpha=0.3)
            plt.xlabel('Epoch')
            plt.ylabel('R²')
            plt.title(f'R² Learning Curves Across Folds ({model_type})')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'learning_curves_{model_type}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # Plot combined model metrics if available
        if any('combined' in r['history'] for r in cv_results):
            plt.figure(figsize=(15, 5))
            for fold_idx, result in enumerate(cv_results):
                if 'combined' in result['history']:
                    plt.plot(result['history']['combined']['val_rmse'],
                            label=f'Fold {fold_idx+1} Val RMSE',
                            alpha=0.3)
                    plt.plot(result['history']['combined']['val_r2'],
                            label=f'Fold {fold_idx+1} Val R²',
                            alpha=0.3)
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.title('Combined Model Validation Metrics Across Folds')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'learning_curves_combined.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
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
        plt.title(f'RMSE Learning Curves Across Folds ({config.model_type})')
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
        plt.title(f'R² Learning Curves Across Folds ({config.model_type})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'learning_curves_{config.model_type}.png'), dpi=300, bbox_inches='tight')
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
    
    # Print dataset sizes before processing
    print(f"\nDataset sizes before processing:")
    print(f"Predicted samples: {len(predicted_df)}")
    print(f"Experimental samples: {len(experimental_df)}")
    
    # Convert to float
    predicted_df['CCS_Predicted'] = predicted_df['CCS_Predicted'].astype(float)
    experimental_df['CCS_Experimental'] = experimental_df['CCS_Experimental'].astype(float)
    
    # Ensure same length
    if len(predicted_df) != len(experimental_df):
        min_len = min(len(predicted_df), len(experimental_df))
        print(f"\nWarning: Trimming datasets to match length of {min_len}")
        predicted_df = predicted_df.iloc[:min_len]
        experimental_df = experimental_df.iloc[:min_len]
    
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
    
    # Create data loaders with TPU-optimized settings
    train_loader = DataLoader(
        dataset_train,
        batch_size=512,  # Reduced batch size
        shuffle=True,
        drop_last=False,  # Changed to False to keep all samples
        num_workers=0  # Disable multiprocessing for TPU
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=512,  # Reduced batch size
        shuffle=False,
        drop_last=False,  # Changed to False to keep all samples
        num_workers=0  # Disable multiprocessing for TPU
    )
    
    print(f"Training samples: {len(dataset_train)}")
    print(f"Test samples: {len(dataset_test)}")
    
    return train_loader, test_loader, normalizer

if __name__ == '__main__':
    main() 
