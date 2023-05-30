import torch.nn as nn
from torchvision import datasets, models

"""
def set_parameter_requires_grad(model, feature_extracting, num_layers=7):
    if feature_extracting:
        child_counter = 0
        for child in model.children():
            print("child ",child_counter)#,child)
            if child_counter < num_layers:
                print("child ",child_counter," was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                child_counter+=1
            else:
                print("child ",child_counter," was not frozen")
                child_counter += 1
"""
"""
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
"""

def set_parameter_requires_grad(model, feature_extracting, num_layers=7):
    if feature_extracting:
        child_counter = 0
        for child in model.children():
            print("child ",child_counter)#,child)
            if child_counter == 0:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < 18:
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                        print("child ", children_of_child_counter, 'of child',child_counter,' was frozen')
                        children_of_child_counter+=1
                    else:
                        print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                        children_of_child_counter += 1
                child_counter += 1
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
        model = models.mobilenet_v2(weights=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
        print(model)
        set_parameter_requires_grad(model,True,8)
        #num_features = model.fc.in_features
        #model.fc = nn.Linear(in_features=num_features,out_features=1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
        return model
    

get_model('fe')

#def set_parameter_requires_grad(model, feature_extracting):
#    if feature_extracting:
#        for param in model.parameters():
#            param.requires_grad = False