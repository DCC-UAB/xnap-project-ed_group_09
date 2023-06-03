import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def get_model(tipus=None):
    if tipus=='finetunning':
        model = models.resnet34(weights=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,out_features=1)
        return model
    else:
        model = models.resnet34(weights=True)
        set_parameter_requires_grad(model,True)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 1))
        return model