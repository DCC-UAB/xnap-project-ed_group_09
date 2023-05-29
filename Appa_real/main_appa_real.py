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
from lectura_appa_real import FaceDataset, mostrar_imagen
from train_appa_real import *
from model_appa_real import *
from torch.utils.data import ConcatDataset

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#%matplotlib inline

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#url= 'http://158.109.8.102/AppaRealAge/appa-real-release.zip'
#datasets.utils.download_and_extract_archive(url, '../AppaRealAge')

"""Creem projecte wandb"""
"""
wandb.init(
    # set the wandb project where this run will be logged
    project="AP_ct_rt34_mse_20",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "Resnet34",
    "dataset": "AppaReal",
    "epochs": 20,
    }
)
"""
custom_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Define las transformaciones de data augmentation
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

data_dir = "/home/alumne/AppaRealAge/appa-real-release"

train_dataset = FaceDataset(data_dir, "train",transform=custom_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, drop_last=True)

print('Train len:',len(train_loader.dataset))

# Crea un nuevo conjunto de datos aplicando las transformaciones de data augmentation
augmented_dataset = ConcatDataset([train_dataset, train_dataset, train_dataset])

# Aplica las transformaciones de data augmentation al conjunto de datos
augmented_dataset.transforms = augmentation_transform

# Crea el dataloader con el nuevo conjunto de datos aumentado
train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

print('Train post aug len:',len(train_loader.dataset))

val_dataset = FaceDataset(data_dir, "valid",transform=custom_transform)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=2, drop_last=False)

print('Valid len:',len(valid_loader.dataset))

model = get_model('fe')
# Send the model to GPU
model = model.to(device)

name_project='AppaReal-First-Executions'
name_run='fe_augmentation'

# Setup the loss fxn
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 20

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

wandb.finish()

ruta_archivo = 'model_1.pth'

# Guarda el modelo en el archivo
torch.save(model.state_dict(), ruta_archivo)

