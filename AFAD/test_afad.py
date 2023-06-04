from model_afad import *
import torch
from PIL import Image
from torchvision import datasets, models, transforms

model=get_model('fe')

model.load_state_dict(torch.load('./model_fe_drop.pth')) #CHANGE FOR YOUR CASE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

image_path = '/home/alumne/xnap-project-ed_group_09/foto_mare.jpeg' #CHANGE FOR YOUR CASE
image = Image.open(image_path)

input_tensor = transforms.ToTensor()(image)

input_batch = input_tensor.unsqueeze(0)


input_batch = input_batch.to(device)


with torch.no_grad():
    output = model(input_batch)

# Obtener la salida de regresión
predicted_value = output.item()

print(f'Predicción: {predicted_value}')