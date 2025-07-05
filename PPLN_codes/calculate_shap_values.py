"""
SHAP Analysis for CCS Prediction Model
=====================================

This script provides SHAP (SHapley Additive exPlanations) analysis for the ensemble CCS prediction model
to understand feature importance and model interpretability.

Usage:
1. Update the data paths and model path in the main() function
2. Run: python shap_analyzer.py
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
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# Model classes from shap_attention_analyzer.py
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

class TrainingConfig:
    def __init__(self):
        self.ensemble_size = 3
        self.dropout_rate = 0.2
        self.hidden_dim = 128
        self.temperature = 1.0

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
        
    def transform(self, seq_features, charge_mass, targets=None):
        seq_features_2d = seq_features.view(-1, seq_features.size(-1))
        seq_norm = torch.FloatTensor(self.seq_scaler.transform(seq_features_2d)).view(seq_features.size())
        cm_norm = torch.FloatTensor(self.charge_mass_scaler.transform(charge_mass))
        
        if targets is not None:
            targets_norm = torch.FloatTensor(self.target_scaler.transform(targets.reshape(-1, 1))).squeeze()
            return seq_norm, cm_norm, targets_norm
        return seq_norm, cm_norm

class SHAPAnalyzer:
    def __init__(self, model_path, data_paths):
        self.model_path = model_path
        self.data_paths = data_paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalizer = None
        self.config = TrainingConfig()
        self.results_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/shap_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model_and_data(self):
        """Load the trained model and prepare data for analysis."""
        print("Loading model and data...")
        self.model = EnhancedEnsembleCCSPredictor(self.config)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        # Debug: print first sequence layer weights
        try:
            if hasattr(self.model, 'models'):
                print("[DEBUG] First sequence layer weights (ensemble, model 0):", self.model.models[0].sequence_processor[0].weight)
            else:
                print("[DEBUG] First sequence layer weights (single model):", self.model.sequence_processor[0].weight)
        except Exception as e:
            print("[DEBUG] Could not print sequence layer weights:", e)
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
        print("Creating SHAP KernelExplainer...")
        background_seq, background_cm = background_data
        # Concatenate sequence and charge/mass features for SHAP
        background = torch.cat([background_seq, background_cm], dim=1)[:n_background]
        background_np = background.cpu().numpy()
        seq_dim = background_seq.shape[1]
        # Wrapper for model prediction
        def model_predict_wrapper(input_data):
            if isinstance(input_data, np.ndarray):
                input_data = torch.FloatTensor(input_data)
            seq_features = input_data[:, :seq_dim].to(self.device)
            charge_mass = input_data[:, seq_dim:].to(self.device)
            with torch.no_grad():
                predictions = self.model(seq_features, charge_mass)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                return predictions.cpu().numpy()
        explainer = shap.KernelExplainer(model_predict_wrapper, background_np)
        return explainer, seq_dim

    def analyze_feature_importance(self, test_data, explainer, seq_dim, n_samples=100):
        print("Analyzing feature importance...")
        test_seq, test_cm, test_ccs = test_data
        if n_samples < len(test_seq):
            indices = np.random.choice(len(test_seq), n_samples, replace=False)
            sample_seq = test_seq[indices]
            sample_cm = test_cm[indices]
            sample_ccs = test_ccs[indices]
        else:
            sample_seq = test_seq
            sample_cm = test_cm
            sample_ccs = test_ccs
        # Debug: print mean and std of sample_seq
        print("[DEBUG] sample_seq mean:", sample_seq.mean().item(), "std:", sample_seq.std().item())
        # Concatenate for SHAP
        sample = torch.cat([sample_seq, sample_cm], dim=1).cpu().numpy()
        print("[DEBUG] sample shape:", sample.shape)
        print("[DEBUG] background shape:", explainer.data.data.shape)
        # Calculate SHAP values
        print("Calculating SHAP values for all features (sequence + charge/mass)...")
        shap_values = explainer.shap_values(sample, nsamples=100)
        # Split SHAP values
        shap_values_seq = shap_values[:, :seq_dim]
        shap_values_cm = shap_values[:, seq_dim:]
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
    
    def _plot_feature_importance_bar(self, shap_values_seq, shap_values_cm):
        """Plot feature importance as a bar chart."""
        # Calculate mean absolute SHAP values
        mean_shap_seq = np.mean(np.abs(shap_values_seq), axis=0)
        mean_shap_cm = np.mean(np.abs(shap_values_cm), axis=0)
        
        # Get top features
        top_seq_indices = np.argsort(mean_shap_seq)[-20:]  # Top 20 sequence features
        top_seq_values = np.asarray(mean_shap_seq[top_seq_indices]).astype(float).flatten()
        
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
        plt.bar(feature_names, np.asarray(mean_shap_cm).astype(float).flatten())
        plt.ylabel("Mean |SHAP Value|")
        plt.title("Charge/Mass Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "feature_importance_charge_mass.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, shap_values_seq, shap_values_cm):
        """Create a comprehensive analysis report."""
        print("Creating comprehensive analysis report...")
        mean_shap_seq = np.mean(np.abs(shap_values_seq), axis=0)
        top_seq_features = np.argsort(mean_shap_seq)[-10:]
        mean_shap_cm = np.mean(np.abs(shap_values_cm), axis=0)
        # Debug: print SHAP values and indices
        print("[DEBUG] mean_shap_seq:", mean_shap_seq)
        print("[DEBUG] top_seq_features:", top_seq_features)
        print("[DEBUG] mean_shap_cm:", mean_shap_cm)
        report_path = os.path.join(self.results_dir, "interpretability_report.txt")
        with open(report_path, 'w') as f:
            f.write("CCS Prediction Model Interpretability Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            # SHAP Analysis Summary
            f.write("1. SHAP ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            # Sequence feature importance
            f.write(f"Top 10 most important sequence features:\n")
            for i, feat_idx in enumerate(top_seq_features):
                value = mean_shap_seq[feat_idx]
                if hasattr(value, "item"):
                    value = value.item()
                f.write(f"  {i+1}. Seq_Feature_{feat_idx}: {value:.6f}\n")
            # Charge/mass feature importance
            f.write(f"\nCharge/Mass feature importance:\n")
            charge_val = mean_shap_cm[0]
            if hasattr(charge_val, "item"):
                charge_val = charge_val.item()
            mass_val = mean_shap_cm[1]
            if hasattr(mass_val, "item"):
                mass_val = mass_val.item()
            f.write(f"  Charge: {charge_val:.6f}\n")
            f.write(f"  Mass: {mass_val:.6f}\n")
            # Model Insights
            f.write("\n\n2. MODEL INSIGHTS\n")
            f.write("-" * 30 + "\n")
            if mean_shap_cm[0] > mean_shap_cm[1]:
                f.write("• Charge appears to be more important than mass for CCS prediction\n")
            else:
                f.write("• Mass appears to be more important than charge for CCS prediction\n")
            f.write(f"• Sequence features show varying importance levels\n")
            top_feat_val = mean_shap_seq[top_seq_features[-1]]
            if hasattr(top_feat_val, "item"):
                top_feat_val = top_feat_val.item()
            f.write(f"• Top sequence feature has importance: {top_feat_val:.6f}\n")
        print(f"Comprehensive report saved to: {report_path}")
    
    def run_full_analysis(self):
        print("Starting comprehensive SHAP analysis...")
        train_loader, test_loader = self.load_model_and_data()
        background_batch = next(iter(train_loader))
        background_cm, background_seq, _ = background_batch
        explainer, seq_dim = self.create_shap_explainer((background_seq, background_cm))
        test_batch = next(iter(test_loader))
        test_cm, test_seq, test_ccs = test_batch
        shap_values_seq, shap_values_cm = self.analyze_feature_importance(
            (test_seq, test_cm, test_ccs), 
            explainer,
            seq_dim,
            n_samples=100
        )
        self.create_comprehensive_report(shap_values_seq, shap_values_cm)
        print(f"\nAnalysis completed! Results saved in: {self.results_dir}")
        print("\nGenerated files:")
        print("- SHAP summary plots")
        print("- Feature importance visualizations")
        print("- Comprehensive analysis report")

def main():
    """Main function to run the interpretability analysis."""
    
    # Data paths (update these to match your actual paths)
    data_paths = {
        'train_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1_new.tsv',
        'test_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1_new.tsv',
        'train_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_trainnewdata_a1000b1gamma0.pt',
        'test_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_testnewdata_a1000b1gamma0.pt'
    }
    
    # Model path (update this to your actual model path)
    model_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/Results_12June25/best_model_ensemble_fold_5.pt'
    
    # Create analyzer and run analysis
    analyzer = SHAPAnalyzer(model_path, data_paths)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 
