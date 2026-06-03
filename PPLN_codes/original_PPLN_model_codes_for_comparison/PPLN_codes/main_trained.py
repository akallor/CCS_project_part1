"""
@author: a-nakai-k

Code for CCS value prediction using trained model ("trainedmodel.pt").
The predicted CCS values for test data is saved as csv file named 'out_test_predictCCS.csv'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np

np.set_printoptions(threshold=np.inf)

# parameters
data_path_test = './data_test.csv'      # path to csv file for test
column_idx_expccs = 2                   # column index of experimental ccs value data in csv file
column_idx_z = 3                        # column index of charge data in csv file
column_idx_mass = 4                     # column index of mass data in csv file
sequence_path_test = './sequenceTensor_a1000b1gamma0_test.pt'   # path to preprocessed sequence data for test
bs = 200                                # batch size

# data preparation
with open(data_path_test) as f:
    reader = csv.reader(f,delimiter = '\t')
    datalist_test = [row for row in reader]
del(datalist_test[0])   # remove label if necessary
sequence_representations_test = torch.load(sequence_path_test)

z_test = []
seq_testl = []
mass_test = []

for i in range(len(datalist_test)):
    z_test.append(float(datalist_test[i][column_idx_z]))
    mass_test.append(float(datalist_test[i][column_idx_mass]))
    seq_testl.append(sequence_representations_test[i])

z_test = torch.tensor(z_test)
seq_test = torch.stack(seq_testl)
mass_test = torch.tensor(mass_test)

dataset_test = TensorDataset(z_test,seq_test,mass_test)
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
def test(data_loader,model,device):
    predictCCS = np.array([])
    with torch.no_grad():
        for batch, (z,seq,mass) in enumerate(data_loader):
            z = z.to(device, dtype=torch.float)
            seq = seq.to(device)
            mass = mass.to(device, dtype=torch.float)

            num_batch = len(z)
            output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))
            output_np = output.to('cpu').detach().numpy().copy().reshape(-1)
            predictCCS = np.append(predictCCS,output_np)
    return predictCCS

# test
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load trained model
    model = CCSpredictor_PretrainedESM().to(device)
    params = torch.load('trainedmodel.pt')
    params2 = model.state_dict()
    params2['fc1.weight'] = params['fc1.weight']
    params2['fc1.bias'] = params['fc1.bias']
    params2['fc2.weight'] = params['fc2.weight']
    params2['fc2.bias'] = params['fc2.bias']
    params2['fc3.weight'] = params['fc3.weight']
    params2['fc3.bias'] = params['fc3.bias']
    params2['fc4.weight'] = params['fc4.weight']
    params2['fc4.bias'] = params['fc4.bias']
    params2['fc5.weight'] = params['fc5.weight']
    params2['fc5.bias'] = params['fc5.bias']
    params2['fc6.weight'] = params['fc6.weight']
    params2['fc6.bias'] = params['fc6.bias']
    params2['fc7.weight'] = params['fc7.weight']
    params2['fc7.bias'] = params['fc7.bias']
    params2['fc8.weight'] = params['fc8.weight']
    params2['fc8.bias'] = params['fc8.bias']
    params2['fc9.weight'] = params['fc9.weight']
    params2['fc9.bias'] = params['fc9.bias']
    params2['fc10.weight'] = params['fc10.weight']
    params2['fc10.bias'] = params['fc10.bias']
    model.load_state_dict(params2)
    model.eval()

    predictCCS = test(test_loader, model, device)

    # save
    print('test_predictCCS：{0}'.format(predictCCS))
    np.savetxt("out_test_predictCCS.csv", predictCCS, delimiter=",")


if __name__ == '__main__':
    main()
