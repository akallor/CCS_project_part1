import os as os

import pandas as pd


data_all = pd.read_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/combined_mhc_data.tsv",sep = "\t")

#Calculate estimated charge

mass = data_all["Mass"]

mz = data_all["m/z"]

proton_mass = 1.0073

# Estimate charge (z)
data_all['z_estimated'] = np.round(mass / (mz - proton_mass), 0)  # round to nearest integer

#Derive the mean invk0 from the invk0 start and end values after splitting the invk0 range column
data_all[['invk0_start', 'invk0_end']] = data_all['1/k0 Range'].str.split('-', expand=True)

data_all['invk0_start'] = data_all['invk0_start'].astype(float)

data_all['invk0_end'] = data_all['invk0_end'].astype(float)

data_all['invk0_exp'] = data_all[['invk0_start', 'invk0_end']].mean(axis=1)

data_all.drop_duplicates(['Peptide','invk0_exp'])

data_all = data_all.drop(['1/k0 Range','invk0_start','invk0_end'],axis = 1).rename(columns = {'invk0_exp':'invK0'})

data_all.to_csv("/content/drive/MyDrive/Colab_CCS_results/MHC_1/combined_mhc_final_data.tsv",sep = "\t",index = 0)
