from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb
from lectura_cacd import CACDDataset
from train_cacd import *
from model_cacd import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/CACD/cacd_train.csv'
VALID_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/CACD/cacd_valid.csv'
TEST_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/CACD/cacd_test.csv'
IMAGE_PATH = '/home/alumne/datasets/CACD2000'
BATCH_SIZE=256

"""Creem projecte wandb"""
"""
wandb.init(
    # set the wandb project where this run will be logged
    project="CACD executions",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "Resnet34",
    "dataset": "CACD",
    "epochs": 175,
    }
)
"""
custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])

train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])

test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform2)

valid_dataset = CACDDataset(csv_path=VALID_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform2)

print('\nLectura Datasets correcte')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4)
print("TRAIN len")
print(len(train_loader.dataset))

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=4)

print("VALID len")
print(len(valid_loader.dataset))

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=256,
                         shuffle=False,
                         num_workers=4)

print('\nDataLoaders correctes')

print('\nEntrenem el model\n')


model = get_model('fe')
# Send the model to GPU
model = model.to(device)

name_project='CACD executions'
name_run='fe_preprocessat'

# Setup the loss fxn
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

#optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

dataloaders_dict = {}
dataloaders_dict['train']=train_loader
dataloaders_dict['val']=valid_loader

# Train and evaluate
model, losses = train_model_mse(model, dataloaders_dict, criterion, optimizer_ft, num_epochs,name_project,name_run,device)

ruta_archivo = 'model_fe_preprocessat.pth'

# Guarda el modelo en el archivo
torch.save(model.state_dict(), ruta_archivo)

wandb.finish()

