
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import numpy as np
import cv2
import random
#label preprocessing-----------------------
def label_fit(ori_img,resized_img,label):
    """
    resize之後label的位置也要跟著變
    這邊是做resize之後的label mapping。
    """
    new_label = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],dtype=type(label[0]))
    h_ratio = resized_img.shape[0]/ori_img.shape[0]
    w_ratio = resized_img.shape[1]/ori_img.shape[1]
    #print("h_ratio,w_ratio",h_ratio,w_ratio)
    for i in range(4):
        new_label[i*2] = label[i*2] * w_ratio
        new_label[i*2+1]= label[i*2+1] * h_ratio
    
    return new_label
def resize(img,resize_size):
    image = img.copy()
    return cv2.resize(image,resize_size)
#padding----------------------------------------------
def padding(img,pad_size, mode = "ru"):

    if mode == "ru":
        pad_h  = pad_size[0]-img.shape[0]
        pad_w = pad_size[1]-img.shape[1]
        padded_image = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), 'constant', constant_values=0)
        return padded_image
    if mode == "mid":
        pass

def cal_limit(image,key_points):
    min_y_to_top = 1000
    min_y_to_bottom = 1000
    min_x_to_left = 1000
    min_x_to_right = 1000
    xlist = [0,2,4,6]
    ylist = [1,3,5,7]
    for x in xlist:
        XtoLeft = key_points[x]-0 
        XtoRight = image.shape[1]-key_points[x]
        if XtoLeft < min_x_to_left:
            min_x_to_left = XtoLeft
        if XtoRight < min_x_to_right:
            min_x_to_right = XtoRight
    for y in ylist:
        YtoTop = key_points[y]-0
        Ytobottom = image.shape[0]-key_points[y]
        if YtoTop < min_y_to_top:
            min_y_to_top = YtoTop
        if Ytobottom < min_y_to_bottom:
            min_y_to_bottom = Ytobottom
    limit_y = min(min_y_to_top,min_y_to_bottom)
    limit_x = min(min_x_to_left,min_x_to_right)
    return limit_x,limit_y

def augresize(image,resize_size,key_points):
    kps = KeypointsOnImage([
            Keypoint(x=key_points[0], y=key_points[1]),
            Keypoint(x=key_points[2], y=key_points[3]),
            Keypoint(x=key_points[4], y=key_points[5]),
            Keypoint(x=key_points[6], y=key_points[7])
        ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Resize({"height": resize_size[0], "width": resize_size[1]})
    ])

    image_aug, kps_aug = seq(image=image, keypoints=kps)
    newlabel = [0,0,0,0,0,0,0,0]
    for i in range(0,8,2):
        #print(i,end=",")
        #print(i/2,end=",")
        newlabel[i] = kps_aug[int(i/2)].x
        newlabel[i+1] = kps_aug[int(i/2)].y

    return image_aug,newlabel

def KeepSizeResize(image,key_points):
    
    limit_x,limit_y = cal_limit(image,key_points)
    
    crop_limit = min(limit_x,limit_y)
    crop_size = int(random.random()*(crop_limit-10))
    kps = KeypointsOnImage([
            Keypoint(x=key_points[0], y=key_points[1]),
            Keypoint(x=key_points[2], y=key_points[3]),
            Keypoint(x=key_points[4], y=key_points[5]),
            Keypoint(x=key_points[6], y=key_points[7])
        ], shape=image.shape)
    aug = iaa.KeepSizeByResize(
        iaa.Crop((crop_size,crop_size), keep_size=False)
    )

    image_aug, kps_aug = aug(image=image, keypoints=kps)
    #print(kps_aug[0].x)
    newlabel = [0,0,0,0,0,0,0,0]
    for i in range(0,8,2):
        #print(i,end=",")
        #print(i/2,end=",")
        newlabel[i] = kps_aug[int(i/2)].x
        newlabel[i+1] = kps_aug[int(i/2)].y

    return image_aug,newlabel
#----------------------------------------------

