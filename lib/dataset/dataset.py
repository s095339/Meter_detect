import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json

from .preprocessing import padding,resize,label_fit
import numpy as np
import cv2

class MeterDataset(Dataset):
    def __init__(self, cfg, transform,target_transform ):
        #read cfg-------------------
        label_file = cfg.DATASET.LABELROOT
        img_dir = cfg.DATASET.DATAROOT
        preprocess = cfg.DATASET.PREPROCESS
        #---------------------------
        self.label_file = label_file
        self.img_dir = img_dir
        self.img_list,self.label_list = self.read_img_list()
        #print(_C.LOSS)
        if preprocess.lower() == "padding":
            self.imgsize = cfg.DATASET.PADDINGSIZE
        else:
            self.imgsize = cfg.DATASET.IMGSIZE
        self.preprocess = eval(f"{cfg.DATASET.PREPROCESS}")
        
        self.transform = transform
        self.target_transform = target_transform
        print("Len of data = ",len(self))

    def read_img_list(self):
        f = open(self.label_file)
        data = json.load(f)
        return data["images"],data["annotations"]
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx]["filename"])
        label_ID = self.img_list[idx]["id"]
        original_img = cv2.imread(img_path)
        
        #print(original_img.shape)
        

        label = np.array( self.label_list[label_ID]['keypoints'] )
        #print(label)

        image = self.preprocess(original_img,self.imgsize)
        if self.preprocess == resize:
            label = label_fit(original_img,image,label)
            
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



if __name__ == '__main__':
    jason_pth = "D:/DL/Meter_detect/data/train/train_GT_keypoints.json"
    f = open(jason_pth)
    data = json.load(f)
    print(len(data))
    print(data["annotations"][0]["keypoints"])
    
