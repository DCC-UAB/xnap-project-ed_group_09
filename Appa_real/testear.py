from model_appa_real import *
import torch
from PIL import Image
from torchvision import datasets, models, transforms

model=get_model('fe')

model.load_state_dict(torch.load('./model_mix_definitiu.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

image_path = '/home/alumne/xnap-project-ed_group_09/foto_mare.jpeg'
image = Image.open(image_path)

input_tensor = transforms.ToTensor()(image)
# Agregar una dimensión extra para representar el lote (batch)
input_batch = input_tensor.unsqueeze(0)

# Mover el tensor de entrada a la GPU si está disponible
input_batch = input_batch.to(device)

# Realizar la predicción
with torch.no_grad():
    output = model(input_batch)

# Obtener la salida de regresión
predicted_value = output.item()

print(f'Predicción: {predicted_value}')