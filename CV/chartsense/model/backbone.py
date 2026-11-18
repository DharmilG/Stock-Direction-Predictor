import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_backbone():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    
    return model, in_features