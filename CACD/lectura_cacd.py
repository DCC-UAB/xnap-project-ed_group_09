import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image



TRAIN_CSV_PATH = './cacd_train.csv'
VALID_CSV_PATH = './cacd_valid.csv'
TEST_CSV_PATH = './cacd_test.csv'
IMAGE_PATH = '/shared_datasets/CACD/centercropped/jpg'

BATCH_SIZE = 256


class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values

        #self.y = df['age'].values això es per si fem cross_entropy
        self.y= df['file'].str.split('_').str[0].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]


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

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=256,
                         shuffle=False,
                         num_workers=4)

print('\nDataLoaders correctes')
