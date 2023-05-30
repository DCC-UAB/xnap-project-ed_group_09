from model_cacd import *
import torch

model=get_model('finetunning')

model.load_state_dict(torch.load('./model_fnetun_cacd.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()
