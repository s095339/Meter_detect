import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json

from .preprocessing import padding,resize,label_fit,KeepSizeResize,augresize
from lib.dataset.data_aug import donothing,imrotate,augshift,mixaug
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
        
        self.imgsize = cfg.DATASET.IMGSIZE
        self.preprocess = augresize

        
        self.transform = transform
        self.target_transform = target_transform
        #灰階----------------------
        self.gray = self.cfg.DATASET.GRAYSCALE 
        print("training data transform:",transform)
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

        #preprocessing
        #把圖片都放大到480 480
        image,label = self.preprocess(original_img,self.imgsize,label)
        image,label = KeepSizeResize(image,label)
        #data augmentation------------------------------
        if self.cfg.DATAAUG.ENABLE:
            ratio = int(self.cfg.DATAAUG.DATARATIO*10)
            ratio_list = [i<ratio for i in range(10)]
            random.shuffle(ratio_list)
        
            if ratio_list[0]:#做
                #print("------aug------")
                auglist = self.cfg.DATAAUG.TYPE
                
                augratio = self.cfg.DATAAUG.AUGRATIO
                l = []
                index = 0
                for i in augratio:
                    for j in range(i):
                        l.append(index)
                    index += 1
                random.shuffle(l)
                #print(f"augtype = {auglist[l[0]]}")
                augmentation = eval(f"{auglist[l[0]]}")
            else:
                augmentation = donothing
            
            image,label = augmentation(image,label)
        #-----------------------------------------------
        
        
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
        image,_ = self.preprocess(original_img,self.imgsize,[0,0,0,0,0,0,0,0])
        if self.transform:
            image = self.transform(image)
        return image


class SupportDatset(Dataset):
    def __init__(self, cfg, transform):
        self.transform = transform
        self.cfg = cfg
        self.dataroot = cfg.SUPTRAIN.DATAROOT
        self.bs = cfg.SUPTRAIN.BS
        self.preprocess = augresize
        self.label = self.cfg.SUPTRAIN.LABEL # true of false
        self.imgsize = cfg.DATASET.IMGSIZE
        self.gray = self.cfg.DATASET.GRAYSCALE 
        if 48 % self.bs !=0  and not self.label:
            raise ValueError("sup train bs必須可被8整除")
        
        #read sup data in data dir----------------------
        self.dirlist = os.listdir(self.dataroot)
        self.imglist = [] #14個集合 每個集合48個圖
        self.labellist = ["none" for i in range(16)]
        print(self.labellist)
        for dir_n in self.dirlist:
            if dir_n =="0" or dir_n =="15":
                print("neglect dir ",dir_n," this dir only has one picture!!!")
                continue
            dirpath = os.path.join(self.dataroot,dir_n)
            label_index = int(dir_n)
            #label of each suptrain set
            self.labellist[label_index] = os.path.join(dirpath,"gauge_0.png")
            print(f"read imgs from {dirpath}...",end = ",")
            imglist = os.listdir(dirpath)
            if not self.label:
                templist = []
                for img in imglist:
                    #print(img)
                    img_path = os.path.join(dirpath,img)
                    templist.append(img_path)
                print(f"{len(templist)} images in {dirpath}")
                random.shuffle(templist)
                self.imglist.append(templist)
            else: #self.label = true
                for img in imglist:
                    img_path = os.path.join(dirpath,img)
                    self.imglist.append(img_path)
        #實際要運用在訓練的list
        if not self.label:
            self.batch_imglist = []
            for List in self.imglist:
                temp = [List[i:i+self.bs] for i in range(0,len(List),self.bs)]
                for c in temp: self.batch_imglist.append(c)
            random.shuffle(self.batch_imglist)
            
        else:
            self.batch_imglist = self.imglist
            #random.shuffle(self.batch_imglist)
            #print(self.batch_imglist)
        #print(f"{len(self.batch_imglist)} sets of sup data in total")
        #print(self.batch_imglist[50])
        #print(len(self.batch_imglist[50]))
        #split data--------------------------------------------
        print("suplabe = ",self.labellist)
        print("sup data transform:",transform)
    def __len__(self):
        if self.label:
            return len(self.batch_imglist)
        else:
            return 14

    def __getitem__(self, idx):
        """
        強迫dataloader 的batch=1
        然後這邊一次抽幾張由cfg決定。
        """
        if not self.label:
            imglist = self.batch_imglist[idx]
            imgstack = []
            
            for imgpth in imglist:
                if self.gray:
                    original_img = cv2.imread(imgpth,cv2.IMREAD_GRAYSCALE)
                else:
                    original_img = cv2.imread(imgpth)

                #original_img = cv2.imread(imgpth)
                image,_ = self.preprocess(original_img,self.imgsize,[0,0,0,0,0,0,0,0])
                img = self.transform(image).unsqueeze(dim = 0)
                imgstack.append(img)
        
            batchdata = torch.cat(imgstack,0)
            return batchdata
        if self.label:
            img_path = self.batch_imglist[idx]
            #print("img = ",img_path)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            image,_ = self.preprocess(img,self.imgsize,[0,0,0,0,0,0,0,0])
            image = self.transform(image)#.unsqueeze(dim = 0)

            temppath = img_path
            labelindex = temppath.replace('\\','/').split('/')[3]
            labelpath = self.labellist[int(labelindex)]
            label = cv2.imread(labelpath,cv2.IMREAD_GRAYSCALE)
            #print("label =",labelpath)
            label,_ = self.preprocess(label,self.imgsize,[0,0,0,0,0,0,0,0])
            label = self.transform(label)#.unsqueeze(dim = 0)
            return image,label


   