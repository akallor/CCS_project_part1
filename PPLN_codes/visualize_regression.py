import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

# Load datasets
predicted_df = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/400_epochs_default_parameters/out_test_predictCCS_mhc1_400epochs.csv", header=None, names=["CCS_Predicted"])
experimental_df = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/test_data/test_mhc1.tsv", sep="\t")
experimental_df = experimental_df[["CCS_Experimental"]]

# Ensure same length
if len(predicted_df) != len(experimental_df):
    raise ValueError("Datasets have different lengths. Check alignment.")

# Combine into one dataframe
df = pd.concat([experimental_df, predicted_df], axis=1)

# Compute regression
X = df["CCS_Experimental"].values.reshape(-1, 1)
y = df["CCS_Predicted"].values
model = LinearRegression()
model.fit(X, y)

# Create a smooth range of sorted experimental values
x_range = np.linspace(df["CCS_Experimental"].min(), df["CCS_Experimental"].max(), 500).reshape(-1, 1)
y_pred_line = model.predict(x_range)

# Compute error metrics
r2 = r2_score(y, model.predict(X))
mae = mean_absolute_error(y, model.predict(X))
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

# Plot
plt.figure(figsize=(8, 6))

# Scatter points
plt.scatter(df["CCS_Experimental"], df["CCS_Predicted"], color='black', s=10, label='Data')

# Regression line (smooth)
plt.plot(x_range, y_pred_line, color='red', label='Regression line')

# Ideal identity line y = x
min_val = min(df["CCS_Experimental"].min(), df["CCS_Predicted"].min())
max_val = max(df["CCS_Experimental"].max(), df["CCS_Predicted"].max())
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='Ideal prediction (y = x)')

# Residual shaded area: ±1 RMSE around regression line
plt.fill_between(x_range.flatten(), y_pred_line - rmse, y_pred_line + rmse,
                 color='gray', alpha=0.2, label='±1 RMSE')

# Labels and styling
plt.xlabel("Experimental CCS")
plt.ylabel("Predicted CCS")
plt.title("Predicted vs Experimental CCS (MHC Class I)")
plt.text(0.95, 0.05, f"R² = {r2:.3f}", ha='right', va='bottom', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'))

sns.despine()
plt.legend()
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Colab_CCS_results/MHC_1/results/400_epochs_default_parameters/ccspred_vs_ccsexp_mhc1_400epochs_fullband.png", dpi=400)

# Metrics
print(f"R² score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
