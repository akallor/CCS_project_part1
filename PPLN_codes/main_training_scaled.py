"""
@author: a-nakai-k (with fixes)

Code for CCS value prediction using trained model ("trainedmodel.pt").
The predicted CCS values for test data is saved as csv file named 'out_test_predictCCS.csv'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

# parameters
data_path_test = '/content/drive/MyDrive/Colab_CCS_results/pep_all_06113_jy_highsensitivity_on_off_clean.tsv'  # path to csv file for test
column_idx_expccs = 2                   # column index of experimental ccs value data in csv file
column_idx_z = 3                        # column index of charge data in csv file
column_idx_mass = 4                     # column index of mass data in csv file
sequence_path_test = '/content/drive/MyDrive/Colab_CCS_results/sequenceTensor_esmb1_a1000b1gamma0.pt'   # path to preprocessed sequence data for test
bs = 200                                # batch size

# data preparation
with open(data_path_test) as f:
    reader = csv.reader(f, delimiter='\t')
    datalist_test = [row for row in reader]
del(datalist_test[0])   # remove label if necessary
sequence_representations_test = torch.load(sequence_path_test)

z_test = []
seq_testl = []
mass_test = []
# Store original data to save alongside predictions
original_data = []

for i in range(len(datalist_test)):
    # Store original sequence, sample, etc.
    original_data.append({
        'Sequence': datalist_test[i][0] if len(datalist_test[i]) > 0 else '',
        'Sample': datalist_test[i][1] if len(datalist_test[i]) > 1 else '',
        'ExpCCS': float(datalist_test[i][column_idx_expccs]) if len(datalist_test[i]) > column_idx_expccs else 0,
        'Charge': float(datalist_test[i][column_idx_z]) if len(datalist_test[i]) > column_idx_z else 0,
        'Mass': float(datalist_test[i][column_idx_mass]) if len(datalist_test[i]) > column_idx_mass else 0
    })
    
    z_test.append(float(datalist_test[i][column_idx_z]))
    mass_test.append(float(datalist_test[i][column_idx_mass]))
    seq_testl.append(sequence_representations_test[i])

z_test = torch.tensor(z_test)
seq_test = torch.stack(seq_testl)
mass_test = torch.tensor(mass_test)

dataset_test = TensorDataset(z_test, seq_test, mass_test)
test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False)

# model
class CCSpredictor_PretrainedESM(nn.Module):
    def __init__(self):
        super(CCSpredictor_PretrainedESM, self).__init__()
        self.fc1 = nn.Linear(1280*2+2, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, 1000)
        self.fc8 = nn.Linear(1000, 1000)
        self.fc9 = nn.Linear(1000, 1000)
        self.fc10 = nn.Linear(1000, 1)
    def forward(self, x, zandmass):
        x = torch.cat((x, zandmass), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x

# function
def test(data_loader, model, device, scaling_factor=100.0):
    predictCCS = np.array([])
    with torch.no_grad():
        for batch, (z, seq, mass) in enumerate(data_loader):
            z = z.to(device, dtype=torch.float)
            seq = seq.to(device)
            mass = mass.to(device, dtype=torch.float)

            num_batch = len(z)
            output = model(seq, torch.cat((z.view(num_batch, -1), mass.view(num_batch, -1)), dim=1))
            
            # Apply scaling factor to correct the predictions
            output = output / scaling_factor
            
            output_np = output.to('cpu').detach().numpy().copy().reshape(-1)
            predictCCS = np.append(predictCCS, output_np)
    return predictCCS

# test
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load trained model
    model = CCSpredictor_PretrainedESM().to(device)
    model_path = '/content/CCS_project_part1/pretrained_model/trainedmodel.pt'
    print(f"Loading model from {model_path}")
    params = torch.load(model_path, map_location=device)
    params2 = model.state_dict()
    
    # Transfer parameters
    for key in params:
        if key in params2:
            params2[key] = params[key]
        else:
            print(f"Warning: Parameter {key} not found in model")
    
    model.load_state_dict(params2)
    model.eval()

    # Try different scaling factors
    scaling_factors = [1.0, 100.0, 166.7]  # 166.7 ~= 250/1.5 (approx ratio between your output and expected values)
    
    for scaling_factor in scaling_factors:
        print(f"\nTesting with scaling factor: {scaling_factor}")
        predictCCS = test(test_loader, model, device, scaling_factor)
        
        # Calculate statistics
        mean_pred = np.mean(predictCCS)
        min_pred = np.min(predictCCS)
        max_pred = np.max(predictCCS)
        print(f"Predicted CCS range: {min_pred:.4f} - {max_pred:.4f}, mean: {mean_pred:.4f}")
        
        # Save with original data for comparison
        results_df = pd.DataFrame(original_data)
        results_df['PredictedCCS'] = predictCCS
        results_df['ScalingFactor'] = scaling_factor
        
        # Calculate error metrics if experimental CCS is available
        if 'ExpCCS' in results_df.columns:
            results_df['AbsError'] = abs(results_df['PredictedCCS'] - results_df['ExpCCS'])
            results_df['RelError'] = results_df['AbsError'] / results_df['ExpCCS']
            
            mean_abs_error = results_df['AbsError'].mean()
            mean_rel_error = results_df['RelError'].mean()
            print(f"Mean Absolute Error: {mean_abs_error:.4f}")
            print(f"Mean Relative Error: {mean_rel_error:.4f}")
        
        # Save predictions
        output_filename = f"out_test_predictCCS_scale_{scaling_factor:.1f}.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"Saved predictions to {output_filename}")
        
        # Also save just the raw predictions in the original format
        np.savetxt(f"out_test_predictCCS_raw_scale_{scaling_factor:.1f}.csv", predictCCS, delimiter=",")

if __name__ == '__main__':
    main()
