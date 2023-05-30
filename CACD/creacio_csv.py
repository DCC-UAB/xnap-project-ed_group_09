
import os
import csv

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


#csv_path = './cacd_test.csv'
#img_dir = '/home/alumne/datasets/CACD2000-centered'
#nuevo_csv_path = './cacd_test_centered.csv'

#crear_csv(csv_path, img_dir, nuevo_csv_path)


csv_path = './cacd_train.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_train_centered.csv'

crear_csv(csv_path, img_dir, nuevo_csv_path)

csv_path = './cacd_valid.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_valid_centered.csv'

crear_csv(csv_path, img_dir, nuevo_csv_path)


