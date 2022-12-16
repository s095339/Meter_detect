

import numpy as np
import cv2
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
#----------------------------------------------
def img_rotate(img,label):
    pass

