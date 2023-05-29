from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from imgaug import augmenters as iaa
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import ConcatDataset


"""
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img
"""
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
        #ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
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


        """
        if self.augment!=0 and self.augment!=1:
          img = Image.open(img_path)
          img=self.transform(img)
          return img,age
        else:
          img = Image.open(img_path)
          img = img.resize((self.img_size, self.img_size))
          img = self.transform(img).astype(np.float32)
          return torch.from_numpy(np.transpose(img, (2, 0, 1))),age
        #fem np.transpose perque torch espera que les imatges siguin (canal, altura, amplada)"""

def mostrar_imagen(dataset, indice):
    ruta_imagen = dataset.x[indice]
    print(ruta_imagen)
    imagen = mpimg.imread(ruta_imagen)
    plt.imshow(imagen)
    plt.axis('off')
    plt.show()