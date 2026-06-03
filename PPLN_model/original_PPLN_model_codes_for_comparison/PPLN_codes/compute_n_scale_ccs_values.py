from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_data = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/pep_all_06113_jy_highsensitivity_on_off.csv",sep = ",")
test_data_clean = test_data[['Sequence','Sample','invk0','Charge','Mass']].drop_duplicates('Sequence')
test_data_clean.to_csv('/content/drive/MyDrive/Colab_CCS_results/pep_all_06113_jy_highsensitivity_on_off_clean.tsv',sep = '\t',index= 0)


#test_data_clean['CCS'] = (test_data_clean['invk0'] * 3000 * test_data_clean['Charge'])/10
#test_data_clean['CCS'].max()
test_data_clean['K0'] = 1 / test_data_clean['invk0']

# Physical constants
e = 1.602e-19         # Elementary charge in C
N0 = 2.686e25         # Number density at STP in m^-3
kB = 1.381e-23        # Boltzmann constant in J/K
T = 298.15            # Temperature in Kelvin (usually room temp)
pi = np.pi

# Buffer gas: assume nitrogen by default unless otherwise
m_ion = test_data_clean['Mass'] * 1.66054e-27  # Convert peptide mass from Da to kg
m_gas = 28.0134 * 1.66054e-27  # Nitrogen mass in kg
mu = (m_ion * m_gas) / (m_ion + m_gas)         # Reduced mass

# Compute CCS using Mason-Schamp formula
test_data_clean['CCS_Mason'] = (
    (3 * test_data_clean['Charge'] * e) / (16 * N0) *
    np.sqrt((2 * pi) / (mu * kB * T)) *
    (1 / test_data_clean['K0'])                # K0 should be in m^2/Vs
)

# Convert CCS from m^2 to Å^2
test_data_clean['CCS_Mason_A2'] = test_data_clean['CCS_Mason'] * 1e20

# Load predictions
test_predictCCS = pd.read_csv("/content/CCS_project_part1/PPLN_codes/out_test_predictCCS.csv", sep='\t', header=None)
test_predictCCS.columns = ['CCS_predicted']

# Select valid rows with non-NaN CCS_Mason_A2
valid_idx = test_data_clean['CCS_Mason_A2'].notna()
X = test_data_clean.loc[valid_idx, 'CCS_Mason_A2'].values.reshape(-1, 1)

# Make sure y has the same indexing
y = test_predictCCS['CCS_predicted'].values[valid_idx]

# Linear fit
model = LinearRegression()
model.fit(X, y)
scaled_CCS = model.predict(X)



# y_true: Experimental CCS (scaled)
# y_pred: Predicted CCS from your model

y_true = test_data_clean.loc[valid_idx, 'CCS_Mason_scaled'].values
y_pred = test_predictCCS['CCS_predicted'].values[valid_idx]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")


plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
plt.xlabel('Experimental CCS (scaled)')
plt.ylabel('Predicted CCS')
plt.title('Predicted vs Experimental CCS')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save scaled CCS
test_data_clean.loc[valid_idx, 'CCS_Mason_scaled'] = scaled_CCS
