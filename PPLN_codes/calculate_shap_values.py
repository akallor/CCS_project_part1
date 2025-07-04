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
        self.results_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/shap_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model_and_data(self):
        """Load the trained model and prepare data for analysis."""
        print("Loading model and data...")
        
        # Load model - REPLACE THIS SECTION
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
                # REPLACE THIS WITH YOUR ACTUAL MODEL PREDICTION
                predictions = self.model(seq_features, charge_mass)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]  # Take only the final predictions
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
    
    def create_comprehensive_report(self, shap_values_seq, shap_values_cm):
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
            
            # Model Insights
            f.write("\n\n2. MODEL INSIGHTS\n")
            f.write("-" * 30 + "\n")
            
            if mean_shap_cm[0] > mean_shap_cm[1]:
                f.write("• Charge appears to be more important than mass for CCS prediction\n")
            else:
                f.write("• Mass appears to be more important than charge for CCS prediction\n")
            
            f.write(f"• Sequence features show varying importance levels\n")
            f.write(f"• Top sequence feature has importance: {mean_shap_seq[top_seq_features[-1]]:.6f}\n")
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run the complete interpretability analysis."""
        print("Starting comprehensive SHAP analysis...")
        
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
        
        # Create comprehensive report
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
        'train_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/train_1_new_charge3.tsv',
        'test_data': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/test_1_new_charge3.tsv',
        'train_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_trainnewdata_charge3a1000b1gamma0.pt',
        'test_sequence': '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/sequenceTensor_testnewdata_charge3a1000b1gamma0.pt'
    }
    
    # Model path (update this to your actual model path)
    model_path = '/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/results/best_model_ensemble_fold_1.pt'
    
    # Create analyzer and run analysis
    analyzer = SHAPAnalyzer(model_path, data_paths)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 
