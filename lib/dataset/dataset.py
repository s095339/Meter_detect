import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json

from .preprocessing import padding,resize,label_fit
from lib.dataset.data_aug import rotate,shift,noise,mirror_flip,donothing
import numpy as np
import cv2

#---
import random #shuffle a list of sup train
class MeterDataset(Dataset):
    def __init__(self, cfg, transform,target_transform ):
        """
        訓練用的dataset
        """
        self.cfg = cfg
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
        #灰階----------------------
        self.gray = self.cfg.DATASET.GRAYSCALE 
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

        if self.gray:
            original_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        else:
            original_img = cv2.imread(img_path)
        label = np.array( self.label_list[label_ID]['keypoints'] )
        #data augmentation------------------------------
        if self.cfg.DATAAUG.ENABLE:
            ratio = int(self.cfg.DATAAUG.DATARATIO*10)
            ratio_list = [i<ratio for i in range(10)]
            random.shuffle(ratio_list)
        
            if ratio_list[0]:#做
                auglist = self.cfg.DATAAUG.TYPE
                augratio = self.cfg.DATAAUG.AUGRATIO
                l = []
                index = 0
                for i in augratio:
                    for j in range(i):
                        l.append(index)
                    index += 1
                random.shuffle(l)
                augmentation = eval(f"{auglist[l[0]]}")
            else:
                augmentation = donothing
            
            original_img,label = augmentation(original_img.copy(),label)
        #-----------------------------------------------
        #把圖片都放大到640 640
        image = self.preprocess(original_img,self.imgsize)
        if self.preprocess == resize:
            label = label_fit(original_img,image,label)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



class testDataset(Dataset):
    def __init__(self, cfg, transform ):
        """
        測試dataset
        """
        #read cfg-------------------
        self.cfg = cfg
        img_dir = cfg.TEST.DATAROOT
        preprocess = cfg.TEST.PREPROCESS
        #---------------------------
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        #print(_C.LOSS)
        if preprocess.lower() == "padding":
            self.imgsize = cfg.TEST.PADDINGSIZE
        else:
            self.imgsize = cfg.TEST.IMGSIZE
        self.preprocess = eval(f"{cfg.TEST.PREPROCESS}")
        self.transform = transform
        self.gray = self.cfg.DATASET.GRAYSCALE
        print("Len of data = ",len(self))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        if self.gray:
            original_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        else:
            original_img = cv2.imread(img_path)
        image = self.preprocess(original_img,self.imgsize)
        if self.transform:
            image = self.transform(image)
        return image

class SupportDatset(Dataset):
    def __init__(self, cfg, transform):
        """
        sup資料的使用(自監督學習)
        只使用資料夾1~14 因為裡面都統一有48個檔案
        每個資料夾的第一個檔案都是擺正的(當成label)，
        後面都是被助教各種摧殘的鬼神圖片。不過md我的GPU不夠用

        每次固定抖出8張圖片
        要設定這個也不是不行
        不然設定一下好了
        """
        self.transform = transform
        self.cfg = cfg
        self.dataroot = cfg.SUPTRAIN.DATAROOT
        self.bs = cfg.SUPTRAIN.BS
        self.preprocess = resize
        self.imgsize = cfg.DATASET.IMGSIZE
        if 48 % self.bs !=0:
            raise ValueError("sup train bs必須可被8整除")
        
        #read sup data in data dir----------------------
        self.dirlist = os.listdir(self.dataroot)
        self.imglist = [] #14個集合 每個集合48個圖
        for dir_n in self.dirlist:
            if dir_n =="0" or dir_n =="15":
                print("neglect dir ",dir_n," this dir only has one picture!!!")
                continue
            dirpath = os.path.join(self.dataroot,dir_n)
            print(f"read imgs from {dirpath}...",end = ",")
            imglist = os.listdir(dirpath)
            templist = []
            for img in imglist:
                #print(img)
                img_path = os.path.join(dirpath,img)
                templist.append(img_path)
            print(f"{len(templist)} images in {dirpath}")
            random.shuffle(templist)
            self.imglist.append(templist)
        #實際要運用在訓練的list
        self.batch_imglist = []
        for List in self.imglist:
            temp = [List[i:i+self.bs] for i in range(0,len(List),self.bs)]
            for c in temp: self.batch_imglist.append(c)
        random.shuffle(self.batch_imglist)
        #print(f"{len(self.batch_imglist)} sets of sup data in total")
        #print(self.batch_imglist[50])
        #print(len(self.batch_imglist[50]))
        #split data--------------------------------------------
        self.gray = self.cfg.DATASET.GRAYSCALE
    def __len__(self):
        return 14

    def __getitem__(self, idx):
        """
        強迫dataloader 的batch=1
        然後這邊一次抽幾張由cfg決定。
        """
        imglist = self.batch_imglist[idx]
        
        
        #print(imglist)
        imgstack = []
        
        for imgpth in imglist:
            if self.gray:
                original_img = cv2.imread(imgpth,cv2.IMREAD_GRAYSCALE)
            else:
                original_img = cv2.imread(imgpth)

            #original_img = cv2.imread(imgpth)
            image = self.preprocess(original_img,self.imgsize)
            img = self.transform(image).unsqueeze(dim = 0)
            imgstack.append(img)
            #print(img.shape)
        #print(len(imgstack))
        batchdata = torch.cat(imgstack,0)
        #print(batchdata.shape)
        #print(imgstack)

        #print(imgbatch.shape)
        #if self.transform:
            
        #print(imgbatch.shape)
        return batchdata