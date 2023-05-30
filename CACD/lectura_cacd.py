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
import dlib
import numpy as np
import cv2

"""""
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
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
        detected = detector(img, 1)

        if len(detected) == 1:  # skip if there are 0 or more than 1 face
            for idx, face in enumerate(detected):
                width = face.right() - face.left()
                height = face.bottom() - face.top()
                tol = 15
                up_down = 5
                diff = height-width

                if(diff > 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                                (face.left()-tol-int(diff/2)):(face.right()+tol+int(diff/2)),
                                :]
                    else:
                        tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                                (face.left()-tol-int((diff-1)/2)):(face.right()+tol+int((diff+1)/2)),
                                :]
                if(diff <= 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top()-tol-int(diff/2)-up_down):(face.bottom()+tol+int(diff/2)-up_down),
                                (face.left()-tol):(face.right()+tol),
                                :]
                    else:
                        tmp = img[(face.top()-tol-int((diff-1)/2)-up_down):(face.bottom()+tol+int((diff+1)/2)-up_down),
                                (face.left()-tol):(face.right()+tol),
                                :]

                try:
                    img = Image.fromarray(np.uint8(tmp))
                    img = img.resize((120, 120), Image.ANTIALIAS)
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)

                    label = self.y[index]

                    return img, label
                    
                except ValueError:
                    if self.transform is not None:
                        img = self.transform(img)

                    label = self.y[index]

                    return img, label
        else:
            
            
            if self.transform is not None:
                img = self.transform(img)

            label = self.y[index]

            return img, label


            

    def __len__(self):
        return self.y.shape[0]
"""" 

"""
class CACDDataset(Dataset):
    #Custom Dataset for loading CACD face images

    def __init__(self,csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values

        self.y = df['age'].values #aixÃ² es per si fem cross_entropy
        #self.y= df['file'].str.split('_').str[0].values
        self.transform = transform

    def __getitem__(self, index):
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
        detected = detector(img, 1)

        if len(detected) == 1:  # skip if there are 0 or more than 1 face
            for idx, face in enumerate(detected):
                width = face.right() - face.left()
                height = face.bottom() - face.top()
                tol = 15
                up_down = 5
                diff = height-width

                if(diff > 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                                (face.left()-tol-int(diff/2)):(face.right()+tol+int(diff/2)),
                                :]
                    else:
                        tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                                (face.left()-tol-int((diff-1)/2)):(face.right()+tol+int((diff+1)/2)),
                                :]
                if(diff <= 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top()-tol-int(diff/2)-up_down):(face.bottom()+tol+int(diff/2)-up_down),
                                (face.left()-tol):(face.right()+tol),
                                :]
                    else:
                        tmp = img[(face.top()-tol-int((diff-1)/2)-up_down):(face.bottom()+tol+int((diff+1)/2)-up_down),
                                (face.left()-tol):(face.right()+tol),
                                :]

                try:
                    img = Image.fromarray(np.uint8(tmp))
                    img = img.resize((120, 120), Image.ANTIALIAS)
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)

                    label = self.y[index]

                    return img, label
                    
                except ValueError:
                    if self.transform is not None:
                        img = self.transform(img)

                    label = self.y[index]

                    return img, label
        else:
            
            
            if self.transform is not None:
                img = self.transform(img)

            label = self.y[index]

            return img, label


            

    def __len__(self):
        return self.y.shape[0]
"""


class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
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



