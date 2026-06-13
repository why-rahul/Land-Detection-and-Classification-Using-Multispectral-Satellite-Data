import torch
import torch.nn as nn
from torchvision import models

class ResNetMS(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetMS, self).__init__()

        
        self.model = models.resnet18(pretrained=True)

        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)