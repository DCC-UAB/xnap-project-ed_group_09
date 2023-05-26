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


class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values

        self.y = df['age'].values #això es per si fem cross_entropy
        #self.y= df['file'].str.split('_').str[0].values
        self.transform = transform

    def __getitem__(self, index):
        try:
            img_path = os.path.join(self.img_dir, self.img_names[index])
            if os.path.exists(img_path):  # Verificar si la imagen existe en el directorio
                img = Image.open(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                label = self.y[index]

                return img, label
            else:
            # Si la imagen no existe, devuelve una tupla vacía
                return None, None
        except:
            pass

    def __len__(self):
        return self.y.shape[0]



