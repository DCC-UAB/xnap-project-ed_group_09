import torch.nn as nn
from torchvision import datasets, models

def set_parameter_requires_grad(model, feature_extracting, num_layers):
    if feature_extracting:
        total_layers = len(list(model.children()))   # Obtener el número total de capas en el modelo
        print(total_layers)
        for idx, param in enumerate(model.parameters()):
            if idx < total_layers - num_layers:  # Congelar los pesos de las capas anteriores a las últimas 4
                param.requires_grad = False
            else:
                param.requires_grad = True
            
def get_model(tipus=None):
    if tipus=='finetunning':
        model = models.resnet34(weights=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,out_features=1)
        return model
    else:
        model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
        print(model)
        set_parameter_requires_grad(model,True,4)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,out_features=1)
        return model
    

model=get_model('fe')

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

print(params_to_update)
print(len(params_to_update))