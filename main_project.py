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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#%matplotlib inline


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import wandb
from lectura_dataset import FaceDataset,mostrar_imagen
from train_mse import *

#url= 'http://158.109.8.102/AppaRealAge/appa-real-release.zip'
#datasets.utils.download_and_extract_archive(url, '../AppaRealAge')

custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])
data_dir = "../AppaRealAge/appa-real-release"

train_dataset = FaceDataset(data_dir, "train",augment=2,transf=custom_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, drop_last=True)

val_dataset = FaceDataset(data_dir, "valid",augment=2,transf=custom_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=2, drop_last=False)

mostrar_imagen(val_dataset,8)
print(val_dataset.y[8])


"""wandb.init(
    # set the wandb project where this run will be logged
    project="appa_real_customtransform_resnet34_mse_15",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "Resnet34",
    "dataset": "ImageNet",
    "epochs": 15,
    }
)
"""