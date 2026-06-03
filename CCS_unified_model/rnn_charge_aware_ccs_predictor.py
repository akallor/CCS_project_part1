#!/usr/bin/env python3
"""
Hybrid ESM-2 + RNN CCS Predictor with Auxiliary Charge Predictor
==================================================================

This module implements a hybrid method combining:
1. ESM-2 embeddings (frozen) - pooled sequence representations
2. Engineered residue features (nK, nR, nH, nD, nE, length, net_basicity)
3. Auxiliary charge predictor (MLP) - learns charge from sequence
4. Charge embedding injection into LSTM initial state
5. Joint training with CCS prediction and charge prediction

Key Features:
- No explicit charge tokens
- Model learns charge from sequence
- Charge embedding initializes LSTM state
- Joint loss: CCS_loss + λ * charge_prediction_loss

Author: Re-engineered CCS Prediction System
Purpose: Hybrid ESM-2 + RNN CCS prediction with learned charge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import os
import json
from datetime import datetime
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class AuxiliaryChargePredictor(nn.Module):
    """
    Auxiliary MLP that predicts charge from [pooled_esm; engineered_features].
    
    This is a small MLP that learns to predict charge_state from the sequence
    representation, without explicit charge tokens.
    """
    
    def __init__(self, 
                 esm_dim: int,
                 engineered_feat_dim: int = 7,
                 hidden_dim: int = 64):
        """
        Initialize auxiliary charge predictor.
        
        Args:
            esm_dim: Dimension of pooled ESM embeddings
            engineered_feat_dim: Dimension of engineered features (default: 7)
            hidden_dim: Hidden dimension of MLP
        """
        super().__init__()
        
        input_dim = esm_dim + engineered_feat_dim
        
        # Classification: predict charge class (1, 2, 3) - Fix 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes for charges 1, 2, 3
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize MLP weights."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, pooled_esm: torch.Tensor, engineered_features: torch.Tensor) -> torch.Tensor:
        """
        Predict charge from sequence features.
        
        Args:
            pooled_esm: Pooled ESM embeddings [batch_size, esm_dim]
            engineered_features: Engineered residue features [batch_size, engineered_feat_dim]
            
        Returns:
            Predicted charge logits [batch_size, 3] (classification)
        """
        # Concatenate features
        charge_input = torch.cat([pooled_esm, engineered_features], dim=1)
        
        # Predict charge (returns logits for 3 classes)
        charge_logits = self.mlp(charge_input)
        
        return charge_logits

class ChargeEmbeddingLayer(nn.Module):
    """
    Converts predicted charge into a learned embedding vector.
    
    This maps the predicted charge (scalar or class) to a vector representation
    that can be used to initialize the LSTM state.
    """
    
    def __init__(self, charge_embedding_dim: int = 32):
        """
        Initialize charge embedding layer.
        
        Args:
            charge_embedding_dim: Dimension of charge embedding vector
        """
        super().__init__()
        
        self.charge_embedding_dim = charge_embedding_dim
        
        # For classification: use embedding table (Fix 2)
        self.charge_embedding = nn.Embedding(3, charge_embedding_dim)  # 3 charge classes
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize charge embedding weights."""
        nn.init.normal_(self.charge_embedding.weight, mean=0.0, std=0.1)
    
    def forward(self, charge_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted charge logits to embedding.
        
        Args:
            charge_logits: Predicted charge logits [batch_size, 3]
            
        Returns:
            Charge embedding [batch_size, charge_embedding_dim]
        """
        # Get class indices from logits (argmax)
        charge_indices = charge_logits.argmax(dim=1)
        charge_emb = self.charge_embedding(charge_indices)
        
        return charge_emb

class HybridESMRNNCCSPredictor(nn.Module):
    """
    Hybrid ESM-2 + RNN CCS predictor with auxiliary charge predictor.
    
    Architecture (Attention-free):
    1. Pooled ESM embeddings (frozen)
    2. Engineered residue features (nK, nR, nH, nD, nE, length, net_basicity)
    3. Auxiliary charge predictor (MLP) - predicts charge from sequence
    4. Charge embedding from predicted charge
    5. Bidirectional LSTM (2 layers, hidden_dim=256) with charge embedding as initial state
       - Forward LSTM sees N→C direction
       - Backward LSTM sees C→N direction
       - Charge embedding initializes h0 and c0 for all layers and directions
    6. Multi-pooling (mean, max, last_hidden) - attention-free aggregation
    7. Residual projection (Linear → LayerNorm → ReLU)
    8. CCS regressor
    
    No attention mechanisms are used. Multi-pooling replaces attention.
    """
    
    def __init__(self, 
                 esm_dim: int = 320,
                 engineered_feat_dim: int = 7,
                 hidden_dim: int = 256,  # Increased to 256 for better capacity
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 bidirectional: bool = True,  # Use bidirectional LSTM
                 dropout: float = 0.3,
                 charge_embedding_dim: int = 32,
                 max_sequence_length: int = 50):
        """
        Initialize hybrid ESM-2 + RNN CCS predictor.
        
        Args:
            esm_dim: Dimension of pooled ESM embeddings
            engineered_feat_dim: Dimension of engineered features
            hidden_dim: RNN hidden dimension (default: 256)
            num_layers: Number of RNN layers (default: 2)
            rnn_type: Type of RNN ('lstm', 'gru')
            bidirectional: Whether to use bidirectional RNN (default: True)
            dropout: Dropout rate
            charge_embedding_dim: Dimension of charge embedding
            max_sequence_length: Maximum sequence length
        """
        super().__init__()
        
        self.esm_dim = esm_dim
        self.engineered_feat_dim = engineered_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        
        # Calculate effective hidden dimension (2x for bidirectional)
        self.effective_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Step 1: Auxiliary charge predictor (classification)
        self.charge_predictor = AuxiliaryChargePredictor(
            esm_dim=esm_dim,
            engineered_feat_dim=engineered_feat_dim,
            hidden_dim=64
        )
        
        # Step 2: Charge embedding layer (classification)
        self.charge_embedder = ChargeEmbeddingLayer(
            charge_embedding_dim=charge_embedding_dim
        )
        
        # Step 3: Project ESM embeddings to sequence tokens for LSTM
        # We need to reconstruct sequence from pooled ESM
        self.esm_to_sequence = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Positional encoding for sequence reconstruction
        self.positional_encoding = nn.Parameter(
            torch.randn(max_sequence_length, hidden_dim) * 0.1
        )
        
        # Step 4: Bidirectional LSTM layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, 
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Step 5: Project charge embedding to LSTM initial state
        # For bidirectional: need 2x dimensions (forward + backward)
        # For each layer: need hidden_dim per direction
        num_directions = 2 if bidirectional else 1
        total_hidden_per_layer = hidden_dim * num_directions
        
        if self.rnn_type == 'lstm':
            # h0 and c0: [num_layers * num_directions * hidden_dim]
            self.charge_to_h0 = nn.Linear(charge_embedding_dim, num_layers * total_hidden_per_layer)
            self.charge_to_c0 = nn.Linear(charge_embedding_dim, num_layers * total_hidden_per_layer)
        else:  # GRU
            self.charge_to_h0 = nn.Linear(charge_embedding_dim, num_layers * total_hidden_per_layer)
        
        # Step 6: Multi-pooling layer (mean, max, last_hidden)
        # Input: [batch_size, seq_len, effective_hidden_dim]
        # Output: [batch_size, effective_hidden_dim * 3] (mean + max + last)
        
        # Step 7: Residual projection before CCS regressor
        # Multi-pooled features: mean_pool + max_pool + last_hidden = 3 * effective_hidden_dim
        pooled_dim = self.effective_hidden_dim * 3
        self.residual_projection = nn.Sequential(
            nn.Linear(pooled_dim, self.effective_hidden_dim),
            nn.LayerNorm(self.effective_hidden_dim),
            nn.ReLU()
        )
        
        # Step 8: Final CCS predictor
        self.ccs_predictor = nn.Sequential(
            nn.Linear(self.effective_hidden_dim, self.effective_hidden_dim // 2),
            nn.LayerNorm(self.effective_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.effective_hidden_dim // 2, self.effective_hidden_dim // 4),
            nn.LayerNorm(self.effective_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.effective_hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize ESM to sequence projection
        for module in self.esm_to_sequence:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize positional encoding
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.1)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # LSTM forget gate bias initialization
                if self.rnn_type == 'lstm' and 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)  # forget gate
        
        # Initialize charge to LSTM state projections
        if self.rnn_type == 'lstm':
            nn.init.xavier_uniform_(self.charge_to_h0.weight)
            nn.init.zeros_(self.charge_to_h0.bias)
            nn.init.xavier_uniform_(self.charge_to_c0.weight)
            nn.init.zeros_(self.charge_to_c0.bias)
        else:
            nn.init.xavier_uniform_(self.charge_to_h0.weight)
            nn.init.zeros_(self.charge_to_h0.bias)
        
        # Initialize residual projection
        for module in self.residual_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize CCS predictor
        for module in self.ccs_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
    def forward(self, 
                pooled_esm: torch.Tensor, 
                engineered_features: torch.Tensor,
                return_charge_pred: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through hybrid model.
        
        Args:
            pooled_esm: Pooled ESM embeddings [batch_size, esm_dim]
            engineered_features: Engineered residue features [batch_size, engineered_feat_dim]
            return_charge_pred: Whether to return predicted charge
            
        Returns:
            CCS predictions [batch_size, 1]
            (optionally) Predicted charge [batch_size, 1] or [batch_size, num_classes]
        """
        batch_size = pooled_esm.size(0)
        
        # Step 1: Predict charge from sequence features (returns logits)
        charge_logits = self.charge_predictor(pooled_esm, engineered_features)
        
        # Step 2: Convert predicted charge logits to embedding
        charge_emb = self.charge_embedder(charge_logits)
        
        # Step 3: Reconstruct sequence from pooled ESM
        # Project ESM to hidden dimension
        esm_projected = self.esm_to_sequence(pooled_esm)  # [batch_size, hidden_dim]
        
        # Create sequence by adding positional encoding
        # Use a fixed sequence length (e.g., 8 steps) to capture sequence information
        seq_len = 8
        sequence_features = []
        for i in range(seq_len):
            pos_encoded = esm_projected + self.positional_encoding[i].unsqueeze(0).expand(batch_size, -1)
            sequence_features.append(pos_encoded)
        
        # Stack to create sequence
        x = torch.stack(sequence_features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Step 4: Initialize LSTM state from charge embedding
        num_directions = 2 if self.bidirectional else 1
        
        if self.rnn_type == 'lstm':
            h0 = self.charge_to_h0(charge_emb)  # [batch_size, num_layers * num_directions * hidden_dim]
            c0 = self.charge_to_c0(charge_emb)  # [batch_size, num_layers * num_directions * hidden_dim]
            
            # Reshape for bidirectional LSTM
            # PyTorch expects: [num_layers * num_directions, batch_size, hidden_dim]
            # Format: [layer0_forward, layer0_backward, layer1_forward, layer1_backward, ...]
            h0 = h0.view(batch_size, self.num_layers, num_directions, self.hidden_dim)
            # Reorder: [num_layers, batch_size, num_directions, hidden_dim]
            h0 = h0.permute(1, 0, 2, 3).contiguous()
            # Reshape to: [num_layers * num_directions, batch_size, hidden_dim]
            h0 = h0.view(self.num_layers * num_directions, batch_size, self.hidden_dim)
            
            c0 = c0.view(batch_size, self.num_layers, num_directions, self.hidden_dim)
            c0 = c0.permute(1, 0, 2, 3).contiguous()
            c0 = c0.view(self.num_layers * num_directions, batch_size, self.hidden_dim)
            
            # Process through bidirectional LSTM
            rnn_output, (hn, cn) = self.rnn(x, (h0, c0))
        else:  # GRU
            h0 = self.charge_to_h0(charge_emb)  # [batch_size, num_layers * num_directions * hidden_dim]
            h0 = h0.view(batch_size, self.num_layers, num_directions, self.hidden_dim)
            h0 = h0.permute(1, 0, 2, 3).contiguous()
            h0 = h0.view(self.num_layers * num_directions, batch_size, self.hidden_dim)
            rnn_output, hn = self.rnn(x, h0)
        
        # Step 5: Multi-pooling (mean, max, last_hidden) - attention-free alternative
        # rnn_output: [batch_size, seq_len, effective_hidden_dim]
        mean_pool = rnn_output.mean(dim=1)  # [batch_size, effective_hidden_dim]
        max_pool = rnn_output.max(dim=1)[0]  # [batch_size, effective_hidden_dim]
        last_hidden = rnn_output[:, -1, :]  # [batch_size, effective_hidden_dim]
        
        # Concatenate multi-pooled features
        pooled = torch.cat([mean_pool, max_pool, last_hidden], dim=1)  # [batch_size, effective_hidden_dim * 3]
        
        # Step 6: Residual projection
        projected = self.residual_projection(pooled)  # [batch_size, effective_hidden_dim]
        
        # Step 7: Final CCS prediction
        ccs_pred = self.ccs_predictor(projected)  # [batch_size, 1]
        
        if return_charge_pred:
            return ccs_pred, charge_logits
        else:
            return ccs_pred

class HybridESMRNNTrainer:
    """
    Trainer for hybrid ESM-2 + RNN CCS predictor with joint loss.
    """
    
    def __init__(self, 
                 model: HybridESMRNNCCSPredictor,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 gradient_clipping: float = 1.0,
                 charge_loss_weight: float = 0.2,
                 ccs_mean: float = 0.0,
                 ccs_std: float = 1.0,
                 device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model: Hybrid ESM-2 + RNN CCS predictor
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            gradient_clipping: Gradient clipping threshold
            charge_loss_weight: Weight for charge prediction loss (λ)
            ccs_mean: Mean for CCS inverse transform
            ccs_std: Std for CCS inverse transform
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model = model
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.charge_loss_weight = charge_loss_weight
        self.ccs_mean = ccs_mean
        self.ccs_std = ccs_std
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        self.gradient_clipping = gradient_clipping
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_ccs_loss': [],
            'val_ccs_loss': [],
            'train_charge_loss': [],
            'val_charge_loss': [],
            'train_r2': [],
            'val_r2': [],
            'train_charge_r2': [],
            'val_charge_r2': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, criterion) -> Tuple[float, float, float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ccs_loss = 0.0
        total_charge_loss = 0.0
        total_r2 = 0.0
        total_charge_r2 = 0.0
        num_batches = 0
        
        for batch_idx, (pooled_esm, engineered_features, charges, ccs_values) in enumerate(train_loader):
            pooled_esm = pooled_esm.to(self.device)
            engineered_features = engineered_features.to(self.device)
            charges = charges.to(self.device)
            ccs_values = ccs_values.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            ccs_pred, charge_logits = self.model(pooled_esm, engineered_features, return_charge_pred=True)
            
            # Calculate losses separately (Fix 4)
            ccs_loss = criterion(ccs_pred.squeeze(), ccs_values)
            charge_loss = F.cross_entropy(charge_logits, charges)  # Classification loss (Fix 2)
            
            # Joint loss
            loss = ccs_loss + self.charge_loss_weight * charge_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            self.optimizer.step()
            
            # Calculate R² on inverse-transformed CCS (Fix 3)
            ccs_pred_raw = ccs_pred.squeeze() * self.ccs_std + self.ccs_mean
            ccs_values_raw = ccs_values * self.ccs_std + self.ccs_mean
            r2 = self._calculate_r2(ccs_pred_raw, ccs_values_raw)
            
            # Charge accuracy (classification)
            charge_r2 = (charge_logits.argmax(dim=1) == charges).float().mean().item()
            
            total_loss += loss.item()
            total_ccs_loss += ccs_loss.item()
            total_charge_loss += charge_loss.item()
            total_r2 += r2
            total_charge_r2 += charge_r2
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ccs_loss = total_ccs_loss / num_batches
        avg_charge_loss = total_charge_loss / num_batches
        avg_r2 = total_r2 / num_batches
        avg_charge_r2 = total_charge_r2 / num_batches
        
        return avg_loss, avg_ccs_loss, avg_charge_loss, avg_r2, avg_charge_r2
    
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float, float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_ccs_loss = 0.0
        total_charge_loss = 0.0
        total_r2 = 0.0
        total_charge_r2 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for pooled_esm, engineered_features, charges, ccs_values in val_loader:
                pooled_esm = pooled_esm.to(self.device)
                engineered_features = engineered_features.to(self.device)
                charges = charges.to(self.device)
                ccs_values = ccs_values.to(self.device)
                
                ccs_pred, charge_logits = self.model(pooled_esm, engineered_features, return_charge_pred=True)
                
                # Calculate losses separately (Fix 4)
                ccs_loss = criterion(ccs_pred.squeeze(), ccs_values)
                charge_loss = F.cross_entropy(charge_logits, charges)  # Classification loss (Fix 2)
                
                loss = ccs_loss + self.charge_loss_weight * charge_loss
                
                # Calculate R² on inverse-transformed CCS (Fix 3)
                ccs_pred_raw = ccs_pred.squeeze() * self.ccs_std + self.ccs_mean
                ccs_values_raw = ccs_values * self.ccs_std + self.ccs_mean
                r2 = self._calculate_r2(ccs_pred_raw, ccs_values_raw)
                
                # Charge accuracy (classification)
                charge_r2 = (charge_logits.argmax(dim=1) == charges).float().mean().item()
                
                total_loss += loss.item()
                total_ccs_loss += ccs_loss.item()
                total_charge_loss += charge_loss.item()
                total_r2 += r2
                total_charge_r2 += charge_r2
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ccs_loss = total_ccs_loss / num_batches
        avg_charge_loss = total_charge_loss / num_batches
        avg_r2 = total_r2 / num_batches
        avg_charge_r2 = total_charge_r2 / num_batches
        
        return avg_loss, avg_ccs_loss, avg_charge_loss, avg_r2, avg_charge_r2
    
    def _calculate_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate R² score."""
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2.item()
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs: int = 100,
              early_stopping_patience: int = 10,
              save_path: str = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("🚀 Starting Hybrid ESM-2 + RNN CCS Training")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_ccs_loss, train_charge_loss, train_r2, train_charge_r2 = self.train_epoch(train_loader, criterion)
            
            # Validation
            val_loss, val_ccs_loss, val_charge_loss, val_r2, val_charge_r2 = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_ccs_loss'].append(train_ccs_loss)
            self.training_history['val_ccs_loss'].append(val_ccs_loss)
            self.training_history['train_charge_loss'].append(train_charge_loss)
            self.training_history['val_charge_loss'].append(val_charge_loss)
            self.training_history['train_r2'].append(train_r2)
            self.training_history['val_r2'].append(val_r2)
            self.training_history['train_charge_r2'].append(train_charge_r2)
            self.training_history['val_charge_r2'].append(val_charge_r2)
            self.training_history['learning_rates'].append(current_lr)
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}, Train Charge R²: {train_charge_r2:.4f}")
                print(f"           Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}, Val Charge R²: {val_charge_r2:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print("=" * 60)
        print("✅ Training completed!")
        
        return self.training_history

class HybridDataset(torch.utils.data.Dataset):
    """Custom dataset for hybrid ESM + engineered features."""
    
    def __init__(self, esm_features, engineered_features, charges, ccs_values):
        self.esm_features = esm_features
        self.engineered_features = engineered_features
        self.charges = charges
        self.ccs_values = ccs_values
    
    def __len__(self):
        return len(self.esm_features)
    
    def __getitem__(self, idx):
        return self.esm_features[idx], self.engineered_features[idx], self.charges[idx], self.ccs_values[idx]

def _extract_ccs_tensor(dataset: Union[torch.utils.data.Dataset, torch.utils.data.Subset]) -> torch.Tensor:
    """
    Helper to extract CCS values as a tensor from a dataset or subset.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        if hasattr(base_dataset, 'ccs_values'):
            return base_dataset.ccs_values[indices]
        values = []
        for idx in indices:
            _, _, _, ccs_val = base_dataset[idx]
            if not isinstance(ccs_val, torch.Tensor):
                ccs_val = torch.tensor(ccs_val, dtype=torch.float32)
            values.append(ccs_val)
        return torch.stack(values)
    else:
        if hasattr(dataset, 'ccs_values'):
            return dataset.ccs_values
        values = []
        for idx in range(len(dataset)):
            _, _, _, ccs_val = dataset[idx]
            if not isinstance(ccs_val, torch.Tensor):
                ccs_val = torch.tensor(ccs_val, dtype=torch.float32)
            values.append(ccs_val)
        return torch.stack(values)

class NormalizedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies CCS normalization on-the-fly."""
    
    def __init__(self, base_dataset: torch.utils.data.Dataset, ccs_mean: float, ccs_std: float):
        self.base_dataset = base_dataset
        self.ccs_mean = ccs_mean
        self.ccs_std = max(ccs_std, 1e-8)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        esm_feat, eng_feat, charge, ccs_value = self.base_dataset[idx]
        if isinstance(ccs_value, torch.Tensor):
            ccs_norm = (ccs_value - self.ccs_mean) / self.ccs_std
        else:
            ccs_norm = torch.tensor((ccs_value - self.ccs_mean) / self.ccs_std, dtype=torch.float32)
        return esm_feat, eng_feat, charge, ccs_norm

def create_data_loaders(esm_features_path: str, 
                       data_path: str,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       normalization_params: Optional[Dict[str, float]] = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Create data loaders for training and validation.
    
    Args:
        esm_features_path: Path to ESM features file (contains 'esm_features' and 'engineered_features')
        data_path: Path to original data file (for charges and CCS values)
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_loader, val_loader, normalization_params)
        normalization_params: dict with 'ccs_mean' and 'ccs_std' for inverse transform
    """
    # Load features
    features_data = torch.load(esm_features_path, weights_only=False)
    esm_features = features_data['esm_features']
    engineered_features = features_data['engineered_features']
    
    print(f"ESM features type: {type(esm_features)}")
    print(f"ESM features count: {len(esm_features)}")
    if len(esm_features) > 0:
        print(f"First ESM feature shape: {esm_features[0].shape}")
    print(f"Engineered features count: {len(engineered_features)}")
    if len(engineered_features) > 0:
        print(f"First engineered feature shape: {engineered_features[0].shape}")
    
    # Load original data for charges and CCS values
    data_df = pd.read_csv(data_path, sep='\t')
    print(f"Data file shape: {data_df.shape}")
    print(f"Data columns: {list(data_df.columns)}")
    
    # Extract charges and raw CCS values
    charges_raw = data_df['Charge'].values
    charges = torch.tensor(charges_raw - 1, dtype=torch.long)  # 0-indexed for classification
    ccs_values_raw = data_df['CCS_Experimental'].values
    ccs_values = torch.tensor(ccs_values_raw, dtype=torch.float32)
    
    # Check for size mismatch and fix it
    num_esm_features = len(esm_features)
    num_data_rows = len(data_df)
    
    print(f"ESM features count: {num_esm_features}")
    print(f"Data rows count: {num_data_rows}")
    
    if num_esm_features != num_data_rows:
        print(f"⚠️  Size mismatch detected! Trimming to smaller size...")
        min_size = min(num_esm_features, num_data_rows)
        
        # Trim both to the same size
        esm_features = esm_features[:min_size]
        engineered_features = engineered_features[:min_size]
        charges = charges[:min_size]
        ccs_values = ccs_values[:min_size]
        
        print(f"✅ Trimmed both to size: {min_size}")
    
    # Convert ESM features to tensor if they're a list
    if isinstance(esm_features, list):
        esm_tensor = torch.stack(esm_features)
    else:
        esm_tensor = esm_features
    
    # Convert engineered features to tensor
    if isinstance(engineered_features, list):
        engineered_tensor = torch.tensor(np.array(engineered_features), dtype=torch.float32)
    else:
        engineered_tensor = torch.tensor(engineered_features, dtype=torch.float32)
    
    # Create dataset
    dataset = HybridDataset(esm_tensor, engineered_tensor, charges, ccs_values)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    if train_split == 0.0:
        if normalization_params is None:
            raise ValueError("Normalization parameters must be provided when train_split=0.0 (test-only).")
        train_dataset = None
        val_dataset = dataset
    elif train_split == 1.0:
        train_dataset = dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Determine normalization parameters (compute from train split if not provided)
    if normalization_params is None:
        if train_dataset is None:
            raise ValueError("Cannot compute normalization parameters without a training split.")
        train_ccs_tensor = _extract_ccs_tensor(train_dataset)
        ccs_mean = float(train_ccs_tensor.mean().item())
        ccs_std = float(train_ccs_tensor.std(unbiased=False).item())
        if ccs_std < 1e-6:
            ccs_std = 1.0
        normalization_params = {'ccs_mean': ccs_mean, 'ccs_std': ccs_std}
        print(f"CCS normalization (train-only) - Mean: {ccs_mean:.4f}, Std: {ccs_std:.4f}")
        print("⚠️  Normalization computed only on training data to prevent data leakage")
    else:
        ccs_mean = float(normalization_params['ccs_mean'])
        ccs_std = float(max(normalization_params['ccs_std'], 1e-8))
        normalization_params = {'ccs_mean': ccs_mean, 'ccs_std': ccs_std}
        print(f"Using provided CCS normalization params - Mean: {ccs_mean:.4f}, Std: {ccs_std:.4f}")
    
    # Wrap datasets with normalization
    def _wrap_with_normalization(ds):
        if ds is None:
            return None
        return NormalizedDataset(ds, ccs_mean, ccs_std)
    
    train_dataset_norm = _wrap_with_normalization(train_dataset)
    val_dataset_norm = _wrap_with_normalization(val_dataset)
    
    # Create data loaders
    if train_dataset_norm is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None
        
    if val_dataset_norm is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset_norm, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    return train_loader, val_loader, normalization_params

class HybridEvaluator:
    """
    Comprehensive evaluator for hybrid ESM-2 + RNN CCS predictor.
    Provides detailed evaluation metrics, visualizations, and result analysis.
    """
    
    def __init__(self, model: HybridESMRNNCCSPredictor, output_dir: str, ccs_mean: float = 0.0, ccs_std: float = 1.0):
        """
        Initialize evaluator.
        
        Args:
            model: Trained hybrid ESM-2 + RNN CCS predictor
            output_dir: Directory to save evaluation results
            ccs_mean: Mean for CCS inverse transform
            ccs_std: Std for CCS inverse transform
        """
        self.model = model
        self.output_dir = output_dir
        self.device = next(model.parameters()).device
        self.ccs_mean = ccs_mean
        self.ccs_std = ccs_std
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(self, test_loader) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        print("🔍 Evaluating Hybrid ESM-2 + RNN CCS Predictor...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_charges = []
        all_predicted_charges = []
        
        with torch.no_grad():
            for pooled_esm, engineered_features, charges, ccs_values in test_loader:
                pooled_esm = pooled_esm.to(self.device)
                engineered_features = engineered_features.to(self.device)
                charges = charges.to(self.device)
                ccs_values = ccs_values.to(self.device)
                
                ccs_pred, charge_logits = self.model(pooled_esm, engineered_features, return_charge_pred=True)
                
                # Ensure output is 1D
                if ccs_pred.dim() > 1:
                    ccs_pred = ccs_pred.squeeze()
                if ccs_values.dim() > 1:
                    ccs_values = ccs_values.squeeze()
                if charges.dim() > 1:
                    charges = charges.squeeze()
                
                # Inverse transform CCS predictions (Fix 3)
                ccs_pred_raw = ccs_pred.cpu().numpy() * self.ccs_std + self.ccs_mean
                ccs_values_raw = ccs_values.cpu().numpy() * self.ccs_std + self.ccs_mean
                
                all_predictions.extend(ccs_pred_raw)
                all_targets.extend(ccs_values_raw)
                # Convert charges back to 1-indexed for display (0->1, 1->2, 2->3)
                all_charges.extend((charges.cpu().numpy() + 1))
                
                # Charge predictions (classification) - convert back to 1-indexed
                all_predicted_charges.extend((charge_logits.argmax(dim=1).cpu().numpy() + 1))
        
        # Convert to numpy arrays
        predictions_array = np.array(all_predictions).flatten()
        targets_array = np.array(all_targets).flatten()
        charges_array = np.array(all_charges).flatten()
        predicted_charges_array = np.array(all_predicted_charges).flatten()
        
        # Calculate evaluation metrics
        rmse_value = np.sqrt(mean_squared_error(targets_array, predictions_array))
        mae_value = mean_absolute_error(targets_array, predictions_array)
        r2_value = r2_score(targets_array, predictions_array)
        
        # Charge prediction metrics
        charge_rmse = np.sqrt(mean_squared_error(charges_array, predicted_charges_array))
        charge_r2 = r2_score(charges_array, predicted_charges_array)
        
        print(f"\n🎯 HYBRID MODEL PERFORMANCE:")
        print(f"CCS Prediction - RMSE: {rmse_value:.4f}, MAE: {mae_value:.4f}, R²: {r2_value:.4f}")
        print(f"Charge Prediction - RMSE: {charge_rmse:.4f}, R²: {charge_r2:.4f}")
        
        # Save detailed predictions
        results_dataframe = pd.DataFrame({
            'Experimental_CCS': targets_array,
            'Predicted_CCS': predictions_array,
            'True_Charge': charges_array,
            'Predicted_Charge': predicted_charges_array,
            'Absolute_Error': np.abs(targets_array - predictions_array),
            'Relative_Error_Percent': np.abs(targets_array - predictions_array) / targets_array * 100,
            'Residual': predictions_array - targets_array
        })
        
        results_file = os.path.join(self.output_dir, 'hybrid_predictions.tsv')
        results_dataframe.to_csv(results_file, sep='\t', index=False)
        print(f"📊 Predictions saved to: {results_file}")
        
        # Generate comprehensive visualizations
        self._create_evaluation_visualizations(targets_array, predictions_array, charges_array, predicted_charges_array)
        
        # Generate training history visualization
        self._create_training_visualizations()
        
        return {
            'rmse': rmse_value,
            'mae': mae_value,
            'r2': r2_value,
            'charge_rmse': charge_rmse,
            'charge_r2': charge_r2,
            'predictions': predictions_array,
            'targets': targets_array,
            'charges': charges_array,
            'predicted_charges': predicted_charges_array,
            'results_df': results_dataframe
        }
    
    def _create_evaluation_visualizations(self, 
                                        targets: np.ndarray, 
                                        predictions: np.ndarray,
                                        charges: np.ndarray,
                                        predicted_charges: np.ndarray):
        """Create comprehensive evaluation visualizations."""
        print("📈 Generating evaluation visualizations...")
        
        # Set up plotting style (preserve original format)
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hybrid ESM-2 + RNN CCS Prediction Model Evaluation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Predicted vs Experimental
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=20, color='blue')
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Experimental CCS')
        axes[0, 0].set_ylabel('Predicted CCS')
        axes[0, 0].set_title('Predicted vs Experimental CCS')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² value
        r2 = r2_score(targets, predictions)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                       transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residual plot
        residuals = predictions - targets
        axes[0, 1].scatter(targets, residuals, alpha=0.6, s=20, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Experimental CCS')
        axes[0, 1].set_ylabel('Residuals (Predicted - Experimental)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residual Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        axes[0, 2].axvline(mean_residual, color='r', linestyle='--', lw=2,
                          label=f'Mean: {mean_residual:.3f}')
        axes[0, 2].axvline(mean_residual + std_residual, color='orange', linestyle='--', lw=2,
                          label=f'+1σ: {mean_residual + std_residual:.3f}')
        axes[0, 2].axvline(mean_residual - std_residual, color='orange', linestyle='--', lw=2,
                          label=f'-1σ: {mean_residual - std_residual:.3f}')
        axes[0, 2].legend()
        
        # 4. Charge prediction accuracy
        axes[1, 0].scatter(charges, predicted_charges, alpha=0.6, s=20, color='purple')
        axes[1, 0].plot([charges.min(), charges.max()], [charges.min(), charges.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('True Charge')
        axes[1, 0].set_ylabel('Predicted Charge')
        axes[1, 0].set_title('Charge Prediction')
        axes[1, 0].grid(True, alpha=0.3)
        
        charge_r2 = r2_score(charges, predicted_charges)
        axes[1, 0].text(0.05, 0.95, f'R² = {charge_r2:.4f}', 
                       transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Error vs Predicted
        absolute_errors = np.abs(residuals)
        axes[1, 1].scatter(predictions, absolute_errors, alpha=0.6, s=20, color='purple')
        axes[1, 1].set_xlabel('Predicted CCS')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Error vs Predicted CCS')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Charge-specific performance
        unique_charges = np.unique(charges)
        charge_r2_scores = []
        
        for charge in unique_charges:
            charge_mask = charges == charge
            if np.sum(charge_mask) > 10:  # Only if enough samples
                charge_targets = targets[charge_mask]
                charge_predictions = predictions[charge_mask]
                charge_r2 = r2_score(charge_targets, charge_predictions)
                charge_r2_scores.append(charge_r2)
            else:
                charge_r2_scores.append(np.nan)
        
        axes[1, 2].bar(unique_charges, charge_r2_scores, alpha=0.7, color='cyan')
        axes[1, 2].set_xlabel('Charge')
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('R² Score by Charge')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, 'hybrid_evaluation_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plots saved to: {plot_file}")
        plt.close()
    
    def _create_training_visualizations(self):
        """Create training history visualizations with proper scaling."""
        print("📈 Generating training history visualizations...")
        
        # Load training history if available
        history_file = os.path.join(self.output_dir, 'training_history.json')
        if not os.path.exists(history_file):
            print("⚠️  Training history not found. Skipping training visualizations.")
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Create training plots with proper scaling
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hybrid ESM-2 + RNN Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. Loss curves (properly scaled)
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        
        # Scale losses appropriately (better scaling)
        scale_factor = 1
        max_loss = max(train_loss.max(), val_loss.max())
        if max_loss > 100:
            scale_factor = 100
            train_loss_scaled = train_loss / scale_factor
            val_loss_scaled = val_loss / scale_factor
            ylabel = f'Loss (×{scale_factor})'
        elif max_loss > 10:
            scale_factor = 10
            train_loss_scaled = train_loss / scale_factor
            val_loss_scaled = val_loss / scale_factor
            ylabel = f'Loss (×{scale_factor})'
        else:
            train_loss_scaled = train_loss
            val_loss_scaled = val_loss
            ylabel = 'Loss'
        
        axes[0, 0].plot(epochs, train_loss_scaled, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_loss_scaled, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel(ylabel)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² curves (ensure they're in [0, 1] range)
        train_r2 = np.array(history['train_r2'])
        val_r2 = np.array(history['val_r2'])
        
        # Clip R² to reasonable range
        train_r2 = np.clip(train_r2, -1, 1)
        val_r2 = np.clip(val_r2, -1, 1)
        
        axes[0, 1].plot(epochs, train_r2, 'b-', label='Training R²', linewidth=2)
        axes[0, 1].plot(epochs, val_r2, 'r-', label='Validation R²', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Training and Validation R²')
        axes[0, 1].set_ylim(-0.5, 1.0)  # Set reasonable y-axis range
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning rate (log scale, properly scaled)
        learning_rates = np.array(history['learning_rates'])
        
        # Ensure learning rates are positive and reasonable
        learning_rates = np.clip(learning_rates, 1e-8, 1e-1)
        
        axes[0, 2].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Loss difference (overfitting indicator) - properly scaled
        loss_diff = val_loss_scaled - train_loss_scaled
        axes[1, 0].plot(epochs, loss_diff, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        scale_factor_display = scale_factor if max_loss > 10 else 1
        axes[1, 0].set_ylabel(f'Validation Loss - Training Loss (×{scale_factor_display})')
        axes[1, 0].set_title('Overfitting Indicator')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. CCS Loss vs Charge Loss (show both train and val)
        train_ccs_loss = np.array(history['train_ccs_loss'])
        train_charge_loss = np.array(history['train_charge_loss'])
        val_ccs_loss = np.array(history['val_ccs_loss'])
        val_charge_loss = np.array(history['val_charge_loss'])
        
        # Scale if needed (better scaling)
        max_ccs_loss = max(train_ccs_loss.max(), train_charge_loss.max(), val_ccs_loss.max(), val_charge_loss.max())
        if max_ccs_loss > 100:
            scale_factor_ccs = 100
            train_ccs_loss_scaled = train_ccs_loss / scale_factor_ccs
            train_charge_loss_scaled = train_charge_loss / scale_factor_ccs
            val_ccs_loss_scaled = val_ccs_loss / scale_factor_ccs
            val_charge_loss_scaled = val_charge_loss / scale_factor_ccs
            ylabel_ccs = f'Loss (×{scale_factor_ccs})'
        elif max_ccs_loss > 10:
            scale_factor_ccs = 10
            train_ccs_loss_scaled = train_ccs_loss / scale_factor_ccs
            train_charge_loss_scaled = train_charge_loss / scale_factor_ccs
            val_ccs_loss_scaled = val_ccs_loss / scale_factor_ccs
            val_charge_loss_scaled = val_charge_loss / scale_factor_ccs
            ylabel_ccs = f'Loss (×{scale_factor_ccs})'
        else:
            train_ccs_loss_scaled = train_ccs_loss
            train_charge_loss_scaled = train_charge_loss
            val_ccs_loss_scaled = val_ccs_loss
            val_charge_loss_scaled = val_charge_loss
            ylabel_ccs = 'Loss'
        
        axes[1, 1].plot(epochs, train_ccs_loss_scaled, 'b-', label='Train CCS Loss', linewidth=2)
        axes[1, 1].plot(epochs, train_charge_loss_scaled, 'r-', label='Train Charge Loss', linewidth=2)
        axes[1, 1].plot(epochs, val_ccs_loss_scaled, 'b--', label='Val CCS Loss', linewidth=2)
        axes[1, 1].plot(epochs, val_charge_loss_scaled, 'r--', label='Val Charge Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel(ylabel_ccs)
        axes[1, 1].set_title('CCS Loss vs Charge Loss (Train & Val)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Charge prediction R²
        train_charge_r2 = np.array(history['train_charge_r2'])
        val_charge_r2 = np.array(history['val_charge_r2'])
        
        # Clip to reasonable range
        train_charge_r2 = np.clip(train_charge_r2, -1, 1)
        val_charge_r2 = np.clip(val_charge_r2, -1, 1)
        
        axes[1, 2].plot(epochs, train_charge_r2, 'b-', label='Training Charge R²', linewidth=2)
        axes[1, 2].plot(epochs, val_charge_r2, 'r-', label='Validation Charge R²', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('Charge Prediction R²')
        axes[1, 2].set_ylim(-0.5, 1.0)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save training plots
        training_plot_file = os.path.join(self.output_dir, 'hybrid_training_history_plots.png')
        plt.savefig(training_plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plots saved to: {training_plot_file}")
        plt.close()

def main():
    """Main function to run hybrid ESM-2 + RNN CCS prediction."""
    print("🧬 HYBRID ESM-2 + RNN CCS PREDICTOR")
    print("=" * 60)
    
    # Configuration
    config = {
        'esm_features_path': '/hpc/shared/uu_immunopeptidomics/ccs_data/esm_features_train_chg1.pt',
        'data_path': '/hpc/shared/uu_immunopeptidomics/ccs_data/train_1_new_charge1_lab.tsv',
        'test_esm_features_path': '/hpc/shared/uu_immunopeptidomics/ccs_data/esm_features_test_chg1.pt',
        'test_data_path': '/hpc/shared/uu_immunopeptidomics/ccs_data/test_1_new_charge1_lab.tsv',
        'output_dir': '/hpc/shared/uu_immunopeptidomics/ccs_data/hybrid_esm_rnn_results',
        'model_config': {
            'esm_dim': 320,
            'engineered_feat_dim': 7,
            'hidden_dim': 256,  # Increased to 256
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,  # Use bidirectional LSTM
            'dropout': 0.1,  # Reduced dropout for closer train/val losses
            'charge_embedding_dim': 32,
            'max_sequence_length': 50
        },
        'training_config': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'early_stopping_patience': 20,
            'charge_loss_weight': 0.2  # λ parameter (Fix 4)
        }
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Check if features exist
    if not os.path.exists(config['esm_features_path']):
        print(f"❌ ESM features not found: {config['esm_features_path']}")
        print("Please run feature extraction first:")
        print("python charge_aware_esm_feature_extraction.py")
        return
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, normalization_params = create_data_loaders(
        config['esm_features_path'],
        config['data_path'],
        batch_size=config['training_config']['batch_size']
    )
    
    # Create model
    print("🏗️  Creating hybrid ESM-2 + RNN model...")
    model = HybridESMRNNCCSPredictor(**config['model_config'])
    
    # Create trainer
    trainer = HybridESMRNNTrainer(
        model,
        learning_rate=config['training_config']['learning_rate'],
        charge_loss_weight=config['training_config']['charge_loss_weight'],
        ccs_mean=normalization_params['ccs_mean'],
        ccs_std=normalization_params['ccs_std']
    )
    
    # Train model
    print("🚀 Training model...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training_config']['num_epochs'],
        early_stopping_patience=config['training_config']['early_stopping_patience'],
        save_path=os.path.join(config['output_dir'], 'best_model.pt')
    )
    
    # Save training history
    with open(os.path.join(config['output_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config['output_dir'], 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Loaded best model for evaluation")
    
    # Create test data loader
    _, test_loader, _ = create_data_loaders(
        config['test_esm_features_path'],
        config['test_data_path'],
        batch_size=config['training_config']['batch_size'],
        train_split=0.0,
        normalization_params=normalization_params
    )
    
    if test_loader is None:
        print("❌ Test loader is None - cannot proceed with evaluation")
        return
    
    # Comprehensive evaluation
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    evaluator = HybridEvaluator(
        model, 
        config['output_dir'],
        ccs_mean=normalization_params['ccs_mean'],
        ccs_std=normalization_params['ccs_std']
    )
    evaluation_results = evaluator.evaluate_model(test_loader)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Architecture: Hybrid ESM-2 + RNN")
    print(f"Final CCS R² Score: {evaluation_results['r2']:.4f}")
    print(f"Final CCS RMSE: {evaluation_results['rmse']:.4f}")
    print(f"Final CCS MAE: {evaluation_results['mae']:.4f}")
    print(f"Final Charge R² Score: {evaluation_results['charge_r2']:.4f}")
    print(f"Final Charge RMSE: {evaluation_results['charge_rmse']:.4f}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training Epochs: {len(history['train_loss'])}")
    print("=" * 60)
    print("🎉 HYBRID ESM-2 + RNN CCS PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()

