import torch
import torch.nn as nn
import torch.functional as F


from torchvision.models import *

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
        #---------------------
        self.arg = arg
        self.cfg = cfg
        #Layer----------------
        self.backbone = eval(f"{self.cfg.MODEL.BACKBONE}()")
        self.decoder = decoder(cfg,arg)
        #print(type(self.backbone))

    def forward(self,x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x