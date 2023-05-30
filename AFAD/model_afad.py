import torch.nn as nn
from torchvision import models
#import pretrainedmodels
#import pretrainedmodels.utils

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
        model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
        set_parameter_requires_grad(model,True)
        num_features = model.fc.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 1))
        return model
    

#model.fc = nn.Sequential(
        #nn.Dropout(p=0.5),
        #nn.Linear(in_features=512, out_features=num_classes)
#    )