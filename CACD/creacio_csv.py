
import os
import csv

def buscar_y_guardar(csv_path, img_dir, nuevo_csv_path):
    # Leer el archivo CSV existente
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Leer la fila de encabezado
        rows = list(reader)  # Leer el resto de filas

    # Obtener la posición de la columna 'file' en el encabezado
    file_index = header.index('file')

    # Filtrar las filas para obtener solo las que tienen archivos existentes en el directorio
    filas_nuevas = []
    for row in rows:
        nombre_archivo = row[file_index]
        ruta_archivo = os.path.join(img_dir, nombre_archivo)
        if os.path.isfile(ruta_archivo):
            filas_nuevas.append(row)

    # Guardar las filas filtradas en un nuevo archivo CSV
    with open(nuevo_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Escribir el encabezado
        writer.writerows(filas_nuevas)  # Escribir las filas filtradas

    print("El nuevo archivo CSV se ha creado correctamente.")

# Ejemplo de uso
csv_path = './cacd_test.csv'
img_dir = '/home/alumne/datasets/CACD2000-centered'
nuevo_csv_path = './cacd_test_centered.csv'

buscar_y_guardar(csv_path, img_dir, nuevo_csv_path)
