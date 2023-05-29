import torch.nn as nn
from torchvision import datasets, models

def set_parameter_requires_grad(model, feature_extracting, num_layers=7):
    if feature_extracting:
        child_counter = 0
        for child in model.children():
            if child_counter < num_layers:
                print("child ",child_counter," was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                child_counter+=1
            else:
                print("child ",child_counter," was not frozen")
                child_counter += 1
            
def get_model(tipus=None):
    if tipus=='finetunning':
        model = models.resnet34(weights=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,out_features=1)
        return model
    else:
        model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
        print(model)
        set_parameter_requires_grad(model,True,7)
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