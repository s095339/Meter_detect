import os
os.mkdir("test")
import torch.nn as nn
import torch
from torchvision.models.resnet import resnet18,resnet34,resnet50
from torchvision.models import *
from config.cfg import _C as cfg
from lib.model.model import MAJIYABAKUNet
import cv2
import numpy



model = MAJIYABAKUNet(cfg)


#prepare data-----------------
os.mkdir("test")
from lib.dataset.dataset import MeterDataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
transform = ToTensor()
target_transform = torch.tensor
dataset = MeterDataset(cfg = cfg,
                        transform = transform,
                        target_transform = target_transform
                        )
trainset = dataset

image,label  = next(iter(trainset))



output = model(image.unsqueeze(axis = 0))

print(output.shape)