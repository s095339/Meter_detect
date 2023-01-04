import torch
import torch.nn as nn
import torch.functional as F


import torchvision.models as models
from .backbone import ResNet34_GRAY,ResidualBlock

class decoder(nn.Module):
    def __init__(self, cfg, arg = None):
        super(decoder, self).__init__()
        #---------------------
        self.arg = arg
        self.cfg = cfg
        #Layer----------------
        
        self.layer = nn.Sequential(
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,8),
            nn.ReLU(),
            
        )

    def forward(self,x):
        x = self.layer(x)
        return x


class MAJIYABAKUNet(nn.Module):
    def __init__(self, cfg, arg = None):
        super(MAJIYABAKUNet, self).__init__()
        self.init_conv = ResidualBlock(1,3)
        #---------------------
        self.arg = arg
        self.cfg = cfg
        #Layer----------------
        self.backbone = models.resnet34(pretrained=True, progress=True)
        
        in_features = self.backbone.fc.in_features
        num_class = 8
        self.backbone.fc = nn.Linear(in_features, num_class) 
                
        #print(type(self.backbone))

    def forward(self,x):
        #print(x.shape)
        x = self.init_conv(x)
        #print(x.shape)
        x = self.backbone(x)
       
        return x