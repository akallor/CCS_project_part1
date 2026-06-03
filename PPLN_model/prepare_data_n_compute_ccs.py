import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
print("Loading data...")
train_data = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/combined_raw_data_sub.tsv", 
                        sep='\t')

#Filter out for MHC-I or II based on requirement

train_data = train_data[train_data.Length <= 12]

print(f"Original data shape: {train_data.shape}")

# Constants for CCS calculation
class CCSConstants:
    e = 1.602176634e-19  # elementary charge in C
    kB = 1.380649e-23    # Boltzmann constant in J/K
    N0 = 2.686763e25     # Loschmidt constant in m⁻³

def convert_invK0_to_CCS(invK0, charge, mass, temperature=298.15, gas_mass=28.013, gas_type="N2"):
    """
    Convert reduced mobility (1/K0) to CCS using Mason-Schamp equation
    
    Parameters:
    invK0 : float or array-like - Reduced mobility in Vs/cm²
    charge : int or array-like - Charge state of the ion
    mass : float or array-like - Mass of the ion in Da
    temperature : float - Temperature in Kelvin (default 298.15 K)
    gas_mass : float - Mass of drift gas in Da (default 28.013 for N2)
    gas_type : str - Type of drift gas ("N2", "He", etc.)
    
    Returns:
    float or array-like - CCS in Å²
    """
    # Correction factors based on gas type
    correction_factors = {"N2": 1.0, "He": 1.36}
    correction = correction_factors.get(gas_type, 1.0)
    
    # Calculate reduced mass (in kg)
    mass_kg = mass * 1.66053906660e-27
    gas_mass_kg = gas_mass * 1.66053906660e-27
    reduced_mass = (mass_kg * gas_mass_kg) / (mass_kg + gas_mass_kg)
    
    # Convert 1/K0 from Vs/cm² to m²/Vs
    invK0_SI = invK0 * 1e-4
    
    # Calculate CCS using Mason-Schamp equation
    CCS = (3 * CCSConstants.e * charge) / (16 * CCSConstants.N0) * \
          np.sqrt(2 * np.pi / (reduced_mass * CCSConstants.kB * temperature)) * \
          invK0_SI * correction
    
    # Convert from m² to Å²
    return CCS * 1e28

# Data cleaning and preprocessing
print("Preprocessing data...")

# Select required columns
required_columns = ['Sample', 'Sequence', 'invk0_exp', 'z_estimated', 'Mass']
data_subset = train_data[required_columns].copy()
data_subset = data_subset.rename(columns = {'z_estimated':'Charge'})

# Remove rows with missing values
data_subset = data_subset.dropna()
print(f"Data shape after removing NaN: {data_subset.shape}")

# Handle duplicate sequences
print("Handling duplicate sequences...")
# Option 1: Keep first occurrence of each sequence
# data_clean = data_subset.drop_duplicates('Sequence', keep='first')

# Option 2: Average measurements for duplicate sequences (recommended)
data_clean = data_subset.groupby('Sequence').agg({
    'Sample': 'first',  # Keep first sample name
    'invk0_exp': 'mean',    # Average invk0 values
    'Charge': 'first',  # Assuming charge is consistent for same sequence
    'Mass': 'first'     # Assuming mass is consistent for same sequence
}).reset_index()

print(f"Data shape after handling duplicates: {data_clean.shape}")

# Calculate CCS for all data
print("Calculating CCS values...")
data_clean['CCS_Experimental'] = convert_invK0_to_CCS(
    data_clean['invk0_exp'],
    data_clean['Charge'], 
    data_clean['Mass']
)

# Check for any invalid CCS values
invalid_ccs = data_clean['CCS_Experimental'].isna() | (data_clean['CCS_Experimental'] <= 0)
if invalid_ccs.any():
    print(f"Warning: Found {invalid_ccs.sum()} invalid CCS values. Removing them.")
    data_clean = data_clean[~invalid_ccs]

# Random train-test split (80-20)
print("Performing random train-test split...")
train_data_split, test_data_split = train_test_split(
    data_clean, 
    test_size=0.2, 
    random_state=42,  # For reproducibility
    stratify=None     # Could stratify by charge if needed: stratify=data_clean['Charge']
)

print(f"Training set size: {len(train_data_split)}")
print(f"Test set size: {len(test_data_split)}")

# Select final columns for output
output_columns = ['Sample', 'Sequence', 'CCS_Experimental', 'Charge', 'Mass']

# Save the splits
print("Saving train and test sets...")
train_data_split[output_columns].to_csv(
    "/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/train_1.tsv", 
    sep='\t', 
    index=False
)

test_data_split[output_columns].to_csv(
    "/content/drive/MyDrive/Colab_CCS_results/MHC_1/processed_data/test_1.tsv", 
    sep='\t', 
    index=False
)

# Print some basic statistics
print("\n=== Summary Statistics ===")
print("Training set CCS statistics:")
print(train_data_split['CCS_Experimental'].describe())
print("\nTest set CCS statistics:")
print(test_data_split['CCS_Experimental'].describe())

# Check charge distribution
print(f"\nCharge distribution in training set:")
print(train_data_split['Charge'].value_counts().sort_index())
print(f"\nCharge distribution in test set:")
print(test_data_split['Charge'].value_counts().sort_index())

print("\nSplit completed successfully!")
