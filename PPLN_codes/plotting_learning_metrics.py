import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the logged metrics
history_df = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/training_metrics.tsv", sep='\t')

# Apply smoothing
sigma = 2  # adjust for smoothing level
for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_evs', 'test_evs']:
    history_df[f'{metric}_smooth'] = gaussian_filter1d(history_df[metric], sigma=sigma)

# Identify best epoch (lowest test RMSE)
best_epoch = history_df['test_rmse'].idxmin()
best_epoch_num = history_df.loc[best_epoch, 'epoch']

# Plot
plt.figure(figsize=(20, 10))

# RMSE
plt.subplot(2, 2, 1)
plt.plot(history_df['epoch'], history_df['train_rmse_smooth'], label='Train RMSE')
plt.plot(history_df['epoch'], history_df['test_rmse_smooth'], label='Test RMSE')
plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE over Epochs')
plt.legend()

# MAE
plt.subplot(2, 2, 2)
plt.plot(history_df['epoch'], history_df['train_mae_smooth'], label='Train MAE')
plt.plot(history_df['epoch'], history_df['test_mae_smooth'], label='Test MAE')
plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE over Epochs')
plt.legend()

# R² Score
plt.subplot(2, 2, 3)
plt.plot(history_df['epoch'], history_df['train_r2_smooth'], label='Train R²')
plt.plot(history_df['epoch'], history_df['test_r2_smooth'], label='Test R²')
plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('R² over Epochs')
plt.legend()

# Explained Variance
plt.subplot(2, 2, 4)
plt.plot(history_df['epoch'], history_df['train_evs_smooth'], label='Train Explained Variance')
plt.plot(history_df['epoch'], history_df['test_evs_smooth'], label='Test Explained Variance')
plt.axvline(x=best_epoch_num, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Explained Variance')
plt.title('Explained Variance over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
