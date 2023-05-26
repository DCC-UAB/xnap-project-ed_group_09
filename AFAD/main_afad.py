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

TRAIN_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_train.csv'
VALID_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_valid.csv'
TEST_CSV_PATH = '/home/alumne/xnap-project-ed_group_09/AFAD/afad_test.csv'
IMAGE_PATH = '/home/alumne/datasets/AFAD_Full'
BATCH_SIZE=256

"""Creem projecte wandb"""

wandb.init(
    # set the wandb project where this run will be logged
    project="AFAD executions",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "Resnet34",
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
                          batch_size=256,
                          shuffle=True,
                          num_workers=4)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=256,
                         shuffle=False,
                         num_workers=4)

print('\nDataLoaders correctes')

print('\nEntrenem el model\n')


model = get_model('feature extraction')
# Send the model to GPU
model = model.to(device)

name_project='AFAD executions'
name_run='feature extraction'

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

ruta_archivo = 'model_feature_extraction.pth'

# Guarda el modelo en el archivo
torch.save(model.state_dict(), ruta_archivo)

wandb.finish()

