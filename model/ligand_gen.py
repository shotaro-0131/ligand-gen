import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import torch
import os

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(6, 100, 3)
        self.conv2 = nn.Conv3d(100, 32, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2)
        self.batch1 = nn.BatchNorm3d(6)
        self.batch2 = nn.BatchNorm3d(100)

    def forward(self, x):
        # x = self.batch1(x)
        x = self.conv1(x)
        # x = self.relu(x)
        # x = self.pool(x)
        # x = self.batch2(x)
        x = self.conv2(x)
        # x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_t1 = nn.ConvTranspose3d(32, 100, 3)
        self.conv_t2 = nn.ConvTranspose3d(100, 6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_t1(x)
        # x = self.relu(x)
        x = self.conv_t2(x)
        # x = self.relu(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        x = self.dec(self.enc(x))
        return x

class DataSet:
    def __init__(self, train, label):
        self.train = train
        self.label = label
    def __len__(self):
        return len(self.train)
    def __getitem__(self, index):
        return self.train[index], self.label[index]

class Trainer:
    def __init__(self, model, optimizer, dataset, criterion, batch_size=50, epoch=10, display=True, auto_save=True):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = DataLoader(dataset,  batch_size = batch_size)
        self.batch_size = batch_size
        self.epoch=epoch
        self.criterion = criterion
        self.display = display
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auto_save = auto_save
    
    def __log(self, epoch=0, loss=0):
        if self.display:
            if epoch == 0:
                print("epoch" + " "*10 + "loss")
            else:
                print(str(epoch) + " "*10 + str(loss))

    def run(self):
        self.__log()
        for e in range(self.epoch):
            self.model.train()
            average_loss = 0.0
            for i, data in enumerate(self.dataloader):
                protein, ligand = data
                protein.to(self.device)
                ligand.to(self.device)
                self.optimizer.zero_grad()
                ligand_pred = self.model(protein)
                loss = self.criterion(ligand, ligand_pred)
                loss.backward()
                self.optimizer.step()
                average_loss += loss.item()
            self.__log(e+1, average_loss/(i+1))
            if self.auto_save:
                self.save_model("temp_model")

    def save_model(self, filename):
        torch.save(self.model, filename)
    
    def get_model(self):
        return self.model


def traverse(root):
    for d, _, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)               


def get_trainer(root):
    
    for i, h5filename in enumerate(traverse(root)):
        with h5py.File(h5filename) as h5:
            if i == 0:
                protein_data = np.array(h5['protein'][:], dtype=np.float32)
                ligand_data = np.array(h5['ligand'][:], dtype=np.float32)
                # ligand_data = np.array(h5['protein'][:], dtype=np.float32)

                continue
            protein_data = np.concatenate([protein_data, np.array(h5['protein'][:], dtype=np.float32)])
            ligand_data = np.concatenate([ligand_data, np.array(h5['ligand'][:], dtype=np.float32)])

    train_data = DataSet(protein_data, ligand_data)
    model = AutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.MSELoss()
    
    return Trainer(model, optimizer, train_data, loss)
