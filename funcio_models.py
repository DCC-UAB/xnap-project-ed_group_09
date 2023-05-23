import torch.nn as nn
#import pretrainedmodels
#import pretrainedmodels.utils

def get_model(num_classes=101, tipus=None):
  if tipus=='classificació':
    # Resnet18 with pretrained weights 
      model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
      model.fc = nn.Linear(in_features=512,out_features=num_classes)
      return model
  else:
    model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    model.fc = nn.Linear(in_features=512,out_features=1)
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
      model.fc = nn.Linear(in_features=512,out_features=num_classes)
      return model
  else:
    model = models.resnet34(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    set_parameter_requires_grad(model,True)
    model.fc = nn.Linear(in_features=512,out_features=1)
    return model