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
from lectura_afad import *
from train_afad import *
from model_afad import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_train.csv' #CHANGE FOR YOUR CASE
VALID_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_valid.csv' #CHANGE FOR YOUR CASE
TEST_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_test.csv' #CHANGE FOR YOUR CASE
IMAGE_PATH = '/home/alumne/datasets/AFAD-Full' #CHANGE FOR YOUR CASE
BATCH_SIZE = 128

#Creem projecte wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="Execució AFAD",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "Resnet18",
    "dataset": "AFAD",
    "epochs": 15,
    }
)

custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])

train_dataset = AFADDatasetAge(csv_path=TRAIN_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.CenterCrop((120, 120)),
                                        transforms.ToTensor()])

valid_dataset = AFADDatasetAge(csv_path=VALID_CSV_PATH,
                              img_dir=IMAGE_PATH,
                              transform=custom_transform2)

test_dataset = AFADDatasetAge(csv_path=TEST_CSV_PATH,
                              img_dir=IMAGE_PATH,
                              transform=custom_transform2)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)



print('\nDataLoaders correctes:')

print('Train len:',len(train_loader.dataset))
print('Valid len:',len(valid_loader.dataset))
print('Test len:',len(test_loader.dataset))

print('\nEntrenem el model\n')


model = get_model('fe')
# Send the model to GPU
model = model.to(device)

name_project='Execució AFAD'
name_run='feature extraction with dropout'

# Setup the loss fxn
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

# Number of epochs to train for 
num_epochs = 15

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

#optimizer_ft = optim.Adam(model.parameters(), lr=0.1)
optimizer_ft = optim.Adam(params_to_update, lr=0.001)



dataloaders_dict = {}
dataloaders_dict['train']=train_loader
dataloaders_dict['val']=valid_loader

# Train and evaluate
model, losses = train_model_mse(model, dataloaders_dict, criterion, optimizer_ft, num_epochs,name_project,name_run,device)

ruta_archivo = 'model_1.pth' #CHANGE FOR YOUR CASE

# Guarda el modelo en el archivo
torch.save(model.state_dict(), ruta_archivo)

wandb.finish()

