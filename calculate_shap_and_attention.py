"""
SHAP and Attention Analysis for CCS Prediction Model
===================================================

This script provides comprehensive interpretability analysis for the ensemble CCS prediction model:
1. SHAP (SHapley Additive exPlanations) analysis to understand feature importance
2. Attention visualization for sequence features
3. Feature attribution analysis
4. Model interpretability insights

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, mean_squared_error
import os
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the model classes from the existing code
# (The model definitions need to be copied here or imported)
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

class ImprovedCCSPredictor(nn.Module):
    def __init__(self, config, esm_dim=1280*2):
        super(ImprovedCCSPredictor, self).__init__()
        
        self.dropout_rate = config.dropout_rate
        hidden_dim = config.hidden_dim
        
        # Input normalization layers
        self.seq_norm = nn.LayerNorm(esm_dim)
        self.cm_norm = nn.LayerNorm(2)
        
        # Sequence processor with attention
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
        
        # Attention mechanism for sequence features
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=self.dropout_rate)
        
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
                
    def forward(self, seq_features, charge_mass, return_attention=False):
        # Apply input normalization
        seq_features = self.seq_norm(seq_features)
        charge_mass = self.cm_norm(charge_mass)
        
        # Process features
        seq_processed = self.sequence_processor(seq_features)
        cm_processed = self.charge_mass_processor(charge_mass)
        
        # Apply attention to sequence features
        seq_processed_reshaped = seq_processed.unsqueeze(0)  # Add sequence dimension for attention
        attn_output, attn_weights = self.attention(seq_processed_reshaped, seq_processed_reshaped, seq_processed_reshaped)
        seq_processed = attn_output.squeeze(0)  # Remove sequence dimension
        
        # Combine features
        x = torch.cat([seq_processed, cm_processed], dim=1)
        
        # Predict
        prediction = self.predictor(x)
        
        if return_attention:
            return prediction, attn_weights
        return prediction

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
    
    def forward(self, seq_features, charge_mass, return_attention=False):
        # Get predictions from all models
        predictions = []
        attention_weights = []
        
        for model in self.models:
            if return_attention:
                pred, attn = model(seq_features, charge_mass, return_attention=True)
                predictions.append(pred)
                attention_weights.append(attn)
            else:
                pred = model(seq_features, charge_mass)
                predictions.append(pred)
        
        # Stack predictions and apply temperature scaling
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, n_models, 1]
        scaled_weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Weighted average of predictions
        weighted_pred = torch.sum(stacked_preds * scaled_weights.view(1, -1, 1), dim=1)
        
        if return_attention:
            return weighted_pred, predictions, attention_weights
        return weighted_pred

class TrainingConfig:
    def __init__(self, model_type='ensemble'):
        self.bs = 256
        self.base_lr = 1e-4
        self.num_epochs = 400
        self.warmup_epochs = 5
        self.patience = 15
        self.accumulation_steps = 1
        self.model_type = model_type
        self.ensemble_size = 3
        self.ensemble_weights = None
        self.temperature = 1.0
        self.combined_weights = {'improved': 0.5, 'ensemble': 0.5}
        self.weight_decay = 1e-4
        self.label_smoothing = 0.0
        self.dropout_rate = 0.2
        self.hidden_dim = 128
        self.num_folds = 5
        self.min_lr = 1e-6
        self.cycle_momentum = False
        self.cycle_decay = 0.95
        self.mixup_alpha = 0.0
        self.gradient_clip_val = 0.5

class FeatureNormalizer:
    def __init__(self):
        self.seq_scaler = StandardScaler()
        self.charge_mass_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def fit(self, seq_features, charge_mass, targets):
        seq_features_2d = seq_features.view(-1, seq_features.size(-1))
        self.seq_scaler.fit(seq_features_2d)
        self.charge_mass_scaler.fit(charge_mass)
        self.target_scaler.fit(targets.reshape(-1, 1))
        
        self.seq_mean = torch.tensor(self.seq_scaler.mean_, dtype=torch.float32)
        self.seq_std = torch.tensor(self.seq_scaler.scale_, dtype=torch.float32)
        self.target_mean = self.target_scaler.mean_[0]
        self.target_std = self.target_scaler.scale_[0]
        
    def transform(self, seq_features, charge_mass, targets=None):
        seq_features_2d = seq_features.view(-1, seq_features.size(-1))
        seq_norm = torch.FloatTensor(self.seq_scaler.transform(seq_features_2d)).view(seq_features.size())
        cm_norm = torch.FloatTensor(self.charge_mass_scaler.transform(charge_mass))
        
        if targets is not None:
            targets_norm = torch.FloatTensor(self.target_scaler.transform(targets.reshape(-1, 1))).squeeze()
            return seq_norm, cm_norm, targets_norm
        return seq_norm, cm_norm
        
    def inverse_transform_targets(self, targets):
        if isinstance(targets, torch.Tensor):
            targets = targets.reshape(-1, 1)
            if targets.device.type != 'cpu':
                targets = targets.cpu()
            targets = targets.numpy()
        else:
            targets = targets.reshape(-1, 1)
        return torch.FloatTensor(self.target_scaler.inverse_transform(targets))

class SHAPAttentionAnalyzer:
    def __init__(self, model_path, data_paths, config):
        self.model_path = model_path
        self.data_paths = data_paths
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalizer = None
        self.results_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/shap_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model_and_data(self):
        """Load the trained model and prepare data for analysis."""
        print("Loading model and data...")
        
        # Load model
        self.model = EnhancedEnsembleCCSPredictor(self.config)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Load data
        train_loader, test_loader, self.normalizer = self._load_data()
        
        return train_loader, test_loader
    
    def _load_data(self):
        """Load and prepare data for analysis."""
        # Load training data
        with open(self.data_paths['train_data']) as f:
            reader = csv.reader(f, delimiter='\t')
            datalist_train = [row for row in reader]
        del(datalist_train[0])  # remove header

        # Load test data
        with open(self.data_paths['test_data']) as f:
            reader = csv.reader(f, delimiter='\t')
            datalist_test = [row for row in reader]
        del(datalist_test[0])   # remove header

        # Load sequence representations
        sequence_representations_train = torch.load(self.data_paths['train_sequence'])
        sequence_representations_test = torch.load(self.data_paths['test_sequence'])

        # Create normalizer
        normalizer = FeatureNormalizer()
        
        # Prepare training data
        train_seq = torch.stack(sequence_representations_train)
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
        from torch.utils.data import TensorDataset, DataLoader
        dataset_train = TensorDataset(train_cm_norm, train_seq_norm, train_ccs_norm)
        dataset_test = TensorDataset(test_cm_norm, test_seq_norm, test_ccs_norm)
        
        # Create data loaders
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=0)
        
        return train_loader, test_loader, normalizer
    
    def create_shap_explainer(self, background_data, n_background=100):
        """Create SHAP explainer for the model."""
        print("Creating SHAP explainer...")
        
        # Prepare background data
        background_seq, background_cm = background_data
        
        # Create a wrapper function for SHAP
        def model_predict_wrapper(input_data):
            """Wrapper function for model prediction that SHAP can use."""
            if isinstance(input_data, np.ndarray):
                input_data = torch.FloatTensor(input_data)
            
            # Reshape input if needed
            if len(input_data.shape) == 2:
                # Assume this is sequence features
                seq_features = input_data.to(self.device)
                # Use mean charge/mass for background
                charge_mass = background_cm[:input_data.shape[0]].to(self.device)
            else:
                # Assume this is charge/mass features
                charge_mass = input_data.to(self.device)
                # Use mean sequence features for background
                seq_features = background_seq[:input_data.shape[0]].to(self.device)
            
            with torch.no_grad():
                predictions = self.model(seq_features, charge_mass)
                return predictions.cpu().numpy()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model_predict_wrapper,
            background_data[0][:n_background].cpu().numpy(),
            link="identity"
        )
        
        return explainer
    
    def analyze_feature_importance(self, test_data, explainer, n_samples=100):
        """Analyze feature importance using SHAP values."""
        print("Analyzing feature importance...")
        
        test_seq, test_cm, test_ccs = test_data
        
        # Sample data for analysis
        if n_samples < len(test_seq):
            indices = np.random.choice(len(test_seq), n_samples, replace=False)
            sample_seq = test_seq[indices]
            sample_cm = test_cm[indices]
            sample_ccs = test_ccs[indices]
        else:
            sample_seq = test_seq
            sample_cm = test_cm
            sample_ccs = test_ccs
        
        # Calculate SHAP values for sequence features
        print("Calculating SHAP values for sequence features...")
        shap_values_seq = explainer.shap_values(
            sample_seq.cpu().numpy(),
            nsamples=100  # Number of background samples to use
        )
        
        # Calculate SHAP values for charge/mass features
        print("Calculating SHAP values for charge/mass features...")
        shap_values_cm = explainer.shap_values(
            sample_cm.cpu().numpy(),
            nsamples=100
        )
        
        # Analyze and visualize results
        self._visualize_shap_analysis(shap_values_seq, shap_values_cm, sample_seq, sample_cm, sample_ccs)
        
        return shap_values_seq, shap_values_cm
    
    def _visualize_shap_analysis(self, shap_values_seq, shap_values_cm, seq_data, cm_data, ccs_data):
        """Visualize SHAP analysis results."""
        print("Creating SHAP visualizations...")
        
        # 1. Summary plot for sequence features
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_seq,
            seq_data.cpu().numpy(),
            feature_names=[f"Seq_Feature_{i}" for i in range(seq_data.shape[1])],
            show=False
        )
        plt.title("SHAP Summary Plot - Sequence Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shap_summary_sequence.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Summary plot for charge/mass features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_cm,
            cm_data.cpu().numpy(),
            feature_names=["Charge", "Mass"],
            show=False
        )
        plt.title("SHAP Summary Plot - Charge/Mass Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shap_summary_charge_mass.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance bar plot
        self._plot_feature_importance_bar(shap_values_seq, shap_values_cm)
        
        # 4. Individual prediction explanations
        self._plot_individual_explanations(shap_values_seq, shap_values_cm, seq_data, cm_data, ccs_data)
    
    def _plot_feature_importance_bar(self, shap_values_seq, shap_values_cm):
        """Plot feature importance as a bar chart."""
        # Calculate mean absolute SHAP values
        mean_shap_seq = np.mean(np.abs(shap_values_seq), axis=0)
        mean_shap_cm = np.mean(np.abs(shap_values_cm), axis=0)
        
        # Get top features
        top_seq_indices = np.argsort(mean_shap_seq)[-20:]  # Top 20 sequence features
        top_seq_values = mean_shap_seq[top_seq_indices]
        
        # Plot sequence feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_seq_indices)), top_seq_values)
        plt.yticks(range(len(top_seq_indices)), [f"Seq_Feature_{i}" for i in top_seq_indices])
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Top 20 Most Important Sequence Features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "feature_importance_sequence.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot charge/mass feature importance
        plt.figure(figsize=(8, 6))
        feature_names = ["Charge", "Mass"]
        plt.bar(feature_names, mean_shap_cm)
        plt.ylabel("Mean |SHAP Value|")
        plt.title("Charge/Mass Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "feature_importance_charge_mass.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_explanations(self, shap_values_seq, shap_values_cm, seq_data, cm_data, ccs_data):
        """Plot individual prediction explanations."""
        # Select a few representative samples
        n_samples = min(5, len(seq_data))
        sample_indices = np.random.choice(len(seq_data), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Waterfall plot for sequence features
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_seq[idx],
                    base_values=0,
                    data=seq_data[idx].cpu().numpy(),
                    feature_names=[f"Seq_Feature_{j}" for j in range(seq_data.shape[1])]
                ),
                show=False
            )
            plt.title(f"Individual Prediction Explanation - Sample {i+1} (Sequence Features)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"waterfall_sequence_sample_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for charge/mass features
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_cm[idx],
                    base_values=0,
                    data=cm_data[idx].cpu().numpy(),
                    feature_names=["Charge", "Mass"]
                ),
                show=False
            )
            plt.title(f"Individual Prediction Explanation - Sample {i+1} (Charge/Mass Features)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"waterfall_charge_mass_sample_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_attention_weights(self, test_data, n_samples=50):
        """Analyze attention weights from the model."""
        print("Analyzing attention weights...")
        
        test_seq, test_cm, test_ccs = test_data
        
        # Sample data for analysis
        if n_samples < len(test_seq):
            indices = np.random.choice(len(test_seq), n_samples, replace=False)
            sample_seq = test_seq[indices]
            sample_cm = test_cm[indices]
        else:
            sample_seq = test_seq
            sample_cm = test_cm
        
        # Get attention weights from all models in ensemble
        attention_weights_all = []
        
        with torch.no_grad():
            for i, model in enumerate(self.model.models):
                print(f"Getting attention weights from model {i+1}...")
                _, attn_weights = model(sample_seq.to(self.device), sample_cm.to(self.device), return_attention=True)
                attention_weights_all.append(attn_weights.cpu())
        
        # Analyze attention patterns
        self._visualize_attention_analysis(attention_weights_all, sample_seq, sample_cm)
        
        return attention_weights_all
    
    def _visualize_attention_analysis(self, attention_weights_all, seq_data, cm_data):
        """Visualize attention analysis results."""
        print("Creating attention visualizations...")
        
        # 1. Average attention weights across ensemble
        avg_attention = torch.mean(torch.stack(attention_weights_all), dim=0)
        
        # 2. Attention heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            avg_attention[0].numpy(),  # First sample
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title("Attention Weight Heatmap (Ensemble Average)")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "attention_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Attention distribution across models
        plt.figure(figsize=(15, 5))
        for i, attn_weights in enumerate(attention_weights_all):
            plt.subplot(1, len(attention_weights_all), i+1)
            sns.heatmap(
                attn_weights[0].numpy(),
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'}
            )
            plt.title(f"Model {i+1} Attention")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "attention_heatmap_individual_models.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Attention weight distribution
        plt.figure(figsize=(12, 6))
        all_weights = torch.cat(attention_weights_all, dim=0).flatten().numpy()
        plt.hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel("Attention Weight")
        plt.ylabel("Frequency")
        plt.title("Distribution of Attention Weights Across Ensemble")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "attention_weight_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Attention variance across ensemble
        attention_variance = torch.var(torch.stack(attention_weights_all), dim=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_variance[0].numpy(),
            cmap='Reds',
            cbar_kws={'label': 'Attention Weight Variance'}
        )
        plt.title("Attention Weight Variance Across Ensemble")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "attention_variance_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, shap_values_seq, shap_values_cm, attention_weights):
        """Create a comprehensive analysis report."""
        print("Creating comprehensive analysis report...")
        
        report_path = os.path.join(self.results_dir, "interpretability_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("CCS Prediction Model Interpretability Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # SHAP Analysis Summary
            f.write("1. SHAP ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            # Sequence feature importance
            mean_shap_seq = np.mean(np.abs(shap_values_seq), axis=0)
            top_seq_features = np.argsort(mean_shap_seq)[-10:]
            f.write(f"Top 10 most important sequence features:\n")
            for i, feat_idx in enumerate(top_seq_features):
                f.write(f"  {i+1}. Seq_Feature_{feat_idx}: {mean_shap_seq[feat_idx]:.6f}\n")
            
            # Charge/mass feature importance
            mean_shap_cm = np.mean(np.abs(shap_values_cm), axis=0)
            f.write(f"\nCharge/Mass feature importance:\n")
            f.write(f"  Charge: {mean_shap_cm[0]:.6f}\n")
            f.write(f"  Mass: {mean_shap_cm[1]:.6f}\n")
            
            # Attention Analysis Summary
            f.write("\n\n2. ATTENTION ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
            attention_stats = {
                'mean': torch.mean(avg_attention).item(),
                'std': torch.std(avg_attention).item(),
                'max': torch.max(avg_attention).item(),
                'min': torch.min(avg_attention).item()
            }
            
            f.write(f"Attention weight statistics:\n")
            f.write(f"  Mean: {attention_stats['mean']:.6f}\n")
            f.write(f"  Std: {attention_stats['std']:.6f}\n")
            f.write(f"  Max: {attention_stats['max']:.6f}\n")
            f.write(f"  Min: {attention_stats['min']:.6f}\n")
            
            # Model Insights
            f.write("\n\n3. MODEL INSIGHTS\n")
            f.write("-" * 30 + "\n")
            
            if mean_shap_cm[0] > mean_shap_cm[1]:
                f.write("• Charge appears to be more important than mass for CCS prediction\n")
            else:
                f.write("• Mass appears to be more important than charge for CCS prediction\n")
            
            f.write(f"• Sequence features show varying importance levels\n")
            f.write(f"• Attention mechanism shows {attention_stats['std']:.4f} standard deviation\n")
            
            if attention_stats['std'] > 0.1:
                f.write("• High attention variance suggests diverse feature focus across ensemble\n")
            else:
                f.write("• Low attention variance suggests consistent feature focus across ensemble\n")
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run the complete interpretability analysis."""
        print("Starting comprehensive SHAP and Attention analysis...")
        
        # Load model and data
        train_loader, test_loader = self.load_model_and_data()
        
        # Get background data for SHAP
        background_batch = next(iter(train_loader))
        background_seq, background_cm = background_batch[1], background_batch[0]
        
        # Create SHAP explainer
        explainer = self.create_shap_explainer((background_seq, background_cm))
        
        # Get test data
        test_batch = next(iter(test_loader))
        test_seq, test_cm, test_ccs = test_batch
        
        # Run SHAP analysis
        shap_values_seq, shap_values_cm = self.analyze_feature_importance(
            (test_seq, test_cm, test_ccs), 
            explainer, 
            n_samples=100
        )
        
        # Run attention analysis
        attention_weights = self.analyze_attention_weights(
            (test_seq, test_cm, test_ccs), 
            n_samples=50
        )
        
        # Create comprehensive report
        self.create_comprehensive_report(shap_values_seq, shap_values_cm, attention_weights)
        
        print(f"\nAnalysis completed! Results saved in: {self.results_dir}")
        print("\nGenerated files:")
        print("- SHAP summary plots")
        print("- Feature importance visualizations")
        print("- Attention heatmaps")
        print("- Individual prediction explanations")
        print("- Comprehensive analysis report")

def main():
    """Main function to run the interpretability analysis."""
    
    # Configuration
    config = TrainingConfig(model_type='ensemble')
    
    # Data paths (update these to match your actual paths)
    data_paths = {
        'train_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1_new_charge3.tsv',
        'test_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1_new_charge3.tsv',
        'train_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_trainnewdata_charge3a1000b1gamma0.pt',
        'test_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_testnewdata_charge3a1000b1gamma0.pt'
    }
    
    # Model path (update this to your actual model path)
    model_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/best_model_ensemble_fold_1.pt'
    
    # Create analyzer and run analysis
    analyzer = SHAPAttentionAnalyzer(model_path, data_paths, config)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 
