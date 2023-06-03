import dlib
import cv2
import os
import numpy as np
from PIL import Image
import csv

print(f'DLIB: {dlib.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')

# Last tested with
# DLIB: 19.22.0
# NumPy: 1.20.2
# OpenCV: 4.5.2

root_path = '/home/alumne/datasets'

orig_path = os.path.join(root_path, 'CACD2000')
out_path = os.path.join(root_path, 'CACD2000-centered')

def preprocessing(orig_path, out_path):
    if not os.path.exists(orig_path):
        raise ValueError(f'Original image path {orig_path} does not exist.')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    detector = dlib.get_frontal_face_detector()
    keep_picture = []


    for picture_name in os.listdir(orig_path):
        img = cv2.imread(os.path.join(orig_path, picture_name))

        detected = detector(img, 1)

        if len(detected) != 1:  # skip if there are 0 or more than 1 face
            continue

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
                tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))

                cv2.imwrite(os.path.join(out_path, picture_name), tmp)
                print(f'Wrote {picture_name}')
                keep_picture.append(picture_name)
            except ValueError:
                print(f'Failed {picture_name}')
                pass
            

preprocessing(orig_path,out_path)

def crear_csv(csv_path, img_dir, nuevo_csv_path):
    
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  
        rows = list(reader)  

    
    file_index = header.index('file')

    
    filas_nuevas = []
    for row in rows:
        nombre_archivo = row[file_index]
        ruta_archivo = os.path.join(img_dir, nombre_archivo)
        if os.path.isfile(ruta_archivo):
            filas_nuevas.append(row)

    
    with open(nuevo_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  
        writer.writerows(filas_nuevas)  

    print("Nou arxiu creat")


csv_path = './cacd_test.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_test_centered.csv'

crear_csv(csv_path, img_dir, nuevo_csv_path)


csv_path = './cacd_train.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_train_centered.csv'

crear_csv(csv_path, img_dir, nuevo_csv_path)

csv_path = './cacd_valid.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_valid_centered.csv'

crear_csv(csv_path, img_dir, nuevo_csv_path)