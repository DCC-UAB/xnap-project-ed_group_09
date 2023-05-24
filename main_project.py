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
from lectura_dataset import FaceDataset,mostrar_imagen
from train_mse import *
from funcio_models import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#%matplotlib inline

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#url= 'http://158.109.8.102/AppaRealAge/appa-real-release.zip'
#datasets.utils.download_and_extract_archive(url, '../AppaRealAge')

"""Creem projecte wandb"""

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

custom_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#custom_transform = transforms.Compose([transforms.Resize((128, 128)),
#                                       transforms.RandomCrop((120, 120)),
#                                       transforms.ToTensor()])

data_dir = "../AppaRealAge/appa-real-release"

train_dataset = FaceDataset(data_dir, "train",augment=2,transf=custom_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, drop_last=True)

val_dataset = FaceDataset(data_dir, "valid",augment=2,transf=custom_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=2, drop_last=False)

model = get_model_fe()
# Send the model to GPU
model = model.to(device)

name_project='AP_ct_rt34_mse_20'
name_run='fe'

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
dataloaders_dict['val']=val_loader

# Train and evaluate
model, losses = train_model_mse(model, dataloaders_dict, criterion, optimizer_ft, num_epochs,name_project,name_run,device)

ruta_archivo = 'model_1.pth'

# Guarda el modelo en el archivo
torch.save(model.state_dict(), ruta_archivo)

wandb.finish()