"""
@author: a-nakai-k

Code for CCS value prediction using preprocessed sequences.
The predicted CCS values for test data is saved as csv file named 'out_test_predictCCS.csv'.
The trained model composed of fully connected layers is saved as 'trainedmodel.pt'.
List 'ccstest' in l.49 is experimental CCS values for test data 
and used only for calculating test loss, not used for CCS prediction.
If you do not have experimental CCS values for test data, 
please set the variable 'istestloss' to be 'False'.
"""

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np

np.set_printoptions(threshold=np.inf)

# parameters
data_path_train = './data_train.csv'    # path to csv file for training
data_path_test = './data_test.csv'      # path to csv file for test
column_idx_expccs = 2                   # column index of experimental ccs value data in csv file
column_idx_z = 3                        # column index of charge data in csv file
column_idx_mass = 4                     # column index of mass data in csv file
sequence_path_train = './sequenceTensor_a1000b1gamma0_train.pt' # path to preprocessed sequence data for training
sequence_path_test = './sequenceTensor_a1000b1gamma0_test.pt'   # path to preprocessed sequence data for test
bs = 200                                # batch size
lr_adam = 0.0003                        # learning rate
num_ep = 400                            # number of epochs
istestloss = True

# data preparation
with open(data_path_train) as f:
    reader = csv.reader(f,delimiter = '\t')
    datalist_train = [row for row in reader]
del(datalist_train[0])  # remove label if necessary
with open(data_path_test) as f:
    reader = csv.reader(f,delimiter = '\t')
    datalist_test = [row for row in reader]
del(datalist_test[0])   # remove label if necessary
sequence_representations_train = torch.load(sequence_path_train)
sequence_representations_test = torch.load(sequence_path_test)

if istestloss:
    ccs_test = []
ccs_train = []
z_test = []
z_train = []
seq_testl = []
seq_trainl = []
mass_test = []
mass_train = []

for i in range(len(datalist_train)):
    ccs_train.append(float(datalist_train[i][column_idx_expccs]))
    z_train.append(float(datalist_train[i][column_idx_z]))
    mass_train.append(float(datalist_train[i][column_idx_mass]))
    seq_trainl.append(sequence_representations_train[i])
for i in range(len(datalist_test)):
    if istestloss:
        ccs_test.append(float(datalist_test[i][column_idx_expccs]))
    z_test.append(float(datalist_test[i][column_idx_z]))
    mass_test.append(float(datalist_test[i][column_idx_mass]))
    seq_testl.append(sequence_representations_test[i])

if istestloss:
    ccs_test = torch.tensor(ccs_test)
ccs_train = torch.tensor(ccs_train)
z_test = torch.tensor(z_test)
z_train = torch.tensor(z_train)
seq_test = torch.stack(seq_testl)
seq_train = torch.stack(seq_trainl)
mass_test = torch.tensor(mass_test)
mass_train = torch.tensor(mass_train)

dataset_train = TensorDataset(z_train,seq_train,mass_train,ccs_train)
if istestloss:
    dataset_test = TensorDataset(z_test,seq_test,mass_test,ccs_test)
else:
    dataset_test = TensorDataset(z_test,seq_test,mass_test)
train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
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

# functions
def train(data_loader,model,optimizer,criterion,device):
    size = len(data_loader.dataset)
    for batch, (z,seq,mass,ccs) in enumerate(data_loader):
        z = z.to(device, dtype=torch.float)
        seq = seq.to(device)
        mass = mass.to(device, dtype=torch.float)
        ccs = ccs.to(device, dtype=torch.float)

        optimizer.zero_grad()
        num_batch = len(z)
        output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))

        loss = criterion(output, ccs.view(num_batch,-1))
        loss.backward()
        optimizer.step()
        if (batch+1) % 100 == 0:
            rmse, current = np.sqrt(loss.item()), (batch+1) * num_batch
            print(f"loss: {rmse:>7f}  [{current:>5d}/{size:>5d}]")
    return np.sqrt(loss.item())

def test(data_loader,model,criterion,device):
    size = len(data_loader.dataset)
    predictCCS = np.array([])
    if istestloss:
        test_loss = 0
        with torch.no_grad():
            for batch, (z,seq,mass,ccs) in enumerate(data_loader):
                z = z.to(device, dtype=torch.float)
                seq = seq.to(device)
                mass = mass.to(device, dtype=torch.float)
                ccs = ccs.to(device, dtype=torch.float)

                num_batch = len(z)
                output = model(seq,torch.cat((z.view(num_batch,-1),mass.view(num_batch,-1)), dim=1))
                output_np = output.to('cpu').detach().numpy().copy().reshape(-1)
                predictCCS = np.append(predictCCS,output_np)

                test_loss += num_batch * criterion(output,ccs.view(num_batch,-1))
        test_loss /= size
        test_loss = np.sqrt(test_loss.item())
        print(f"Test Error: \n RMSE: {test_loss:>8f} \n")
        return test_loss, predictCCS
    else:
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

# training and test
def main():
    history = {
        'train_rmse': [],
        'test_rmse': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CCSpredictor_PretrainedESM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_adam)

    for epoch in range(num_ep):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # training
        train_loss = train(train_loader, model, optimizer, criterion, device)
        history['train_rmse'].append(train_loss)

        # test
        if istestloss:
            test_loss, predictCCS = test(test_loader, model, criterion, device)
            history['test_rmse'].append(test_loss)
        else:
            predictCCS = test(test_loader, model, criterion, device)

    # save
    torch.save(model.to('cpu').state_dict(),'trainedmodel.pt')
    print(history)
    print('test_predictCCS：{0}'.format(predictCCS))
    np.savetxt("out_test_predictCCS.csv", predictCCS, delimiter=",")


if __name__ == '__main__':
    main()
