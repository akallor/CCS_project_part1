import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression

# Paths to your prediction files
model_files = {
    "IM2Deep": "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/IM2Deep_predictions.tsv",
    "Alphapept": "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/processed_data/Alphapept_predictions.tsv"
}

# Output directory for plots
plot_dir = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/Experiment/ccs_model_comparison_plots"
os.makedirs(plot_dir, exist_ok=True)

def plot_scatter_with_regression(df, plot_dir, model_type):
    X = df["CCS_Experimental"].values.reshape(-1, 1)
    y = df["CCS_Predicted"].values
    model = LinearRegression()
    model.fit(X, y)
    x_range = np.linspace(df["CCS_Experimental"].min(), df["CCS_Experimental"].max(), 500).reshape(-1, 1)
    y_pred_line = model.predict(x_range)
    r2 = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    evs = explained_variance_score(y, model.predict(X))
    slope = model.coef_[0]
    intercept = model.intercept_
    plt.figure(figsize=(10, 8))
    plt.scatter(df["CCS_Experimental"], df["CCS_Predicted"], color='black', s=10, label='Data')
    plt.plot(x_range, y_pred_line, color='red', label=f'Regression line (y = {slope:.3f}x + {intercept:.3f})')
    min_val = min(df["CCS_Experimental"].min(), df["CCS_Predicted"].min())
    max_val = max(df["CCS_Experimental"].max(), df["CCS_Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='Ideal prediction (y = x)')
    plt.fill_between(x_range.flatten(), y_pred_line - rmse, y_pred_line + rmse, color='gray', alpha=0.2, label='±1 RMSE')
    plt.xlabel("Experimental CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"Predicted vs Experimental CCS ({model_type})")
    stats_text = (f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nEVS = {evs:.3f}\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}")
    plt.text(0.05, 0.95, stats_text, ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'scatter_regression_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'scatter_regression_{model_type}.pdf'), dpi=400)
    plt.close()
    return r2, mae, rmse

def plot_residuals(df, plot_dir, model_type):
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    mean_residual = df['Residuals'].mean()
    std_residual = df['Residuals'].std()
    max_residual = df['Residuals'].max()
    min_residual = df['Residuals'].min()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df['CCS_Experimental'], y=df['Residuals'], color='black')
    plt.axhline(0, color='red', linestyle='--', label='Zero line')
    plt.axhline(mean_residual, color='blue', linestyle='--', label='Mean residual')
    plt.xlabel('Experimental CCS')
    plt.ylabel('Residuals (Predicted - Experimental)')
    plt.title(f'Residual Plot: {model_type}')
    stats_text = (f"Mean = {mean_residual:.3f}\nStd Dev = {std_residual:.3f}\nMax = {max_residual:.3f}\nMin = {min_residual:.3f}")
    plt.text(0.05, 0.95, stats_text, ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'residuals_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'residuals_{model_type}.pdf'), dpi=400)
    plt.close()

def plot_residuals_vs_predicted(df, plot_dir, model_type):
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    mean_residual = df['Residuals'].mean()
    std_residual = df['Residuals'].std()
    max_residual = df['Residuals'].max()
    min_residual = df['Residuals'].min()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df['CCS_Predicted'], y=df['Residuals'], color='black')
    plt.axhline(0, color='red', linestyle='--', label='Zero line')
    plt.axhline(mean_residual, color='blue', linestyle='--', label='Mean residual')
    plt.xlabel('Predicted CCS')
    plt.ylabel('Residuals (Predicted - Experimental)')
    plt.title(f'Residual vs Predicted Plot: {model_type}')
    stats_text = (f"Mean = {mean_residual:.3f}\nStd Dev = {std_residual:.3f}\nMax = {max_residual:.3f}\nMin = {min_residual:.3f}")
    plt.text(0.05, 0.95, stats_text, ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'residuals_vs_predicted_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'residuals_vs_predicted_{model_type}.pdf'), dpi=400)
    plt.close()

def plot_overlapping_residuals(df, plot_dir, model_type):
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    mean_residual = df['Residuals'].mean()
    std_residual = df['Residuals'].std()
    max_residual = df['Residuals'].max()
    min_residual = df['Residuals'].min()
    skewness = df['Residuals'].skew()
    kurtosis = df['Residuals'].kurtosis()
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['CCS_Experimental'], y=df['Residuals'], color='blue', alpha=0.5, label='vs Experimental')
    sns.scatterplot(x=df['CCS_Predicted'], y=df['Residuals'], color='black', alpha=0.5, label='vs Predicted')
    plt.axhline(0, color='red', linestyle='--', label='Zero line')
    plt.axhline(mean_residual, color='green', linestyle='--', label='Mean residual')
    plt.xlabel('CCS Value')
    plt.ylabel('Residuals (Predicted - Experimental)')
    plt.title(f'Overlapping Residual Plots: {model_type}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x='Residuals', color='lightblue', fill=True, alpha=0.5, label='Distribution')
    plt.axvline(0, color='red', linestyle='--', label='Zero line')
    plt.axvline(mean_residual, color='green', linestyle='--', label='Mean')
    plt.xlabel('Residuals (Predicted - Experimental)')
    plt.title(f'Residual Distribution: {model_type}')
    stats_text = (f"Distribution Statistics:\nMean = {mean_residual:.3f}\nStd Dev = {std_residual:.3f}\nMax = {max_residual:.3f}\nMin = {min_residual:.3f}\nSkewness = {skewness:.3f}\nKurtosis = {kurtosis:.3f}")
    plt.text(0.95, 0.95, stats_text, ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'overlapping_residuals_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'overlapping_residuals_{model_type}.pdf'), dpi=400)
    plt.close()

def plot_calibration_curve(df, plot_dir, model_type):
    r2 = r2_score(df['CCS_Experimental'], df['CCS_Predicted'])
    mae = mean_absolute_error(df['CCS_Experimental'], df['CCS_Predicted'])
    rmse = np.sqrt(mean_squared_error(df['CCS_Experimental'], df['CCS_Predicted']))
    evs = explained_variance_score(df['CCS_Experimental'], df['CCS_Predicted'])
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df['CCS_Experimental'], y=df['CCS_Predicted'], color='black')
    min_val = min(df['CCS_Experimental'].min(), df['CCS_Predicted'].min())
    max_val = max(df['CCS_Experimental'].max(), df['CCS_Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Identity line')
    plt.xlabel('Experimental CCS')
    plt.ylabel('Predicted CCS')
    plt.title(f'Calibration Curve: {model_type}')
    stats_text = (f"Model Performance:\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nEVS = {evs:.3f}")
    plt.text(0.05, 0.95, stats_text, ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'calibration_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'calibration_{model_type}.pdf'), dpi=400)
    plt.close()

def plot_residual_distribution(df, plot_dir, model_type):
    df['Residuals'] = df['CCS_Predicted'] - df['CCS_Experimental']
    mean_residual = df['Residuals'].mean()
    std_residual = df['Residuals'].std()
    median_residual = df['Residuals'].median()
    skewness = df['Residuals'].skew()
    kurtosis = df['Residuals'].kurtosis()
    q1 = df['Residuals'].quantile(0.25)
    q3 = df['Residuals'].quantile(0.75)
    iqr = q3 - q1
    plt.figure(figsize=(10, 8))
    sns.histplot(df['Residuals'], bins=30, kde=True, color='gray')
    plt.axvline(0, color='red', linestyle='--', label='Zero line')
    plt.axvline(mean_residual, color='blue', linestyle='--', label='Mean')
    plt.axvline(median_residual, color='green', linestyle='--', label='Median')
    plt.xlabel('Residuals (Predicted - Experimental)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Residuals: {model_type}')
    stats_text = (f"Distribution Statistics:\nMean = {mean_residual:.3f}\nMedian = {median_residual:.3f}\nStd Dev = {std_residual:.3f}\nSkewness = {skewness:.3f}\nKurtosis = {kurtosis:.3f}\nQ1 = {q1:.3f}\nQ3 = {q3:.3f}\nIQR = {iqr:.3f}")
    plt.text(0.95, 0.95, stats_text, ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'residual_distribution_{model_type}.png'), dpi=400)
    plt.savefig(os.path.join(plot_dir, f'residual_distribution_{model_type}.pdf'), dpi=400)
    plt.close()

def compute_metrics_and_plots(df, model_type, plot_dir):
    y_true = df["CCS_Experimental"].values
    y_pred = df["predicted_ccs"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_type} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    plot_df = pd.DataFrame({
        "CCS_Experimental": y_true,
        "CCS_Predicted": y_pred
    })
    plot_scatter_with_regression(plot_df, plot_dir, model_type)
    plot_residuals(plot_df, plot_dir, model_type)
    plot_residuals_vs_predicted(plot_df, plot_dir, model_type)
    plot_overlapping_residuals(plot_df, plot_dir, model_type)
    plot_calibration_curve(plot_df, plot_dir, model_type)
    plot_residual_distribution(plot_df, plot_dir, model_type)

def main():
    for model_type, file_path in model_files.items():
        df = pd.read_csv(file_path, sep='\t')
        df = df.drop_duplicates(subset=["peptide_sequences_original"])
        compute_metrics_and_plots(df, model_type, plot_dir)

if __name__ == "__main__":
    main() 
