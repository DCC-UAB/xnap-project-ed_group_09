import torch.nn as nn
from torchvision import datasets, models


def set_parameter_requires_grad(model, feature_extracting, num_layers=7):
    if feature_extracting:
        child_counter = 0
        for child in model.children():
            print("child ",child_counter,child)
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
        model = models.resnet34(weights=True)
        print(model)
        set_parameter_requires_grad(model,True,7)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,out_features=1)
        return model