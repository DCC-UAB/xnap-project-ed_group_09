from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from imgaug import augmenters as iaa
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import ConcatDataset

class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224,transform=None):
        assert(data_type in ("train", "valid", "test"))
        pathcsv = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.transform = transform
            
        self.x = []
        self.y = []
        df = pd.read_csv(str(pathcsv))
        ignore_img_names = list(pd.read_csv('./ignore_list.csv')['img_name'].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["real_age"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]
        img = Image.open(img_path)
        if self.transform is not None:
          img = self.transform(img)
        return img,age