import torch.nn as nn
from torchvision import datasets, models
#import pretrainedmodels
#import pretrainedmodels.utils

def get_model(num_classes=101, tipus=None):
  if tipus=='classificació':
    # Resnet18 with pretrained weights 
      model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
      num_features = model.fc.in_features
      model.fc = nn.Linear(in_features=num_features,out_features=num_classes)
      return model
  else:
    model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features,out_features=1)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model_fe(num_classes=101, tipus=None):
  if tipus=='classificació':
    # Resnet18 with pretrained weights 
      model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
      set_parameter_requires_grad(model,True)
      num_features = model.fc.in_features
      model.fc = nn.Linear(in_features=num_features,out_features=num_classes)
      return model
  else:
    model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    set_parameter_requires_grad(model,True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features,out_features=1)
    return model