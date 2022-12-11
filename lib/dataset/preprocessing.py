

import numpy as np
import cv2
#-----------------------
def padding_transform(img,pad_size, mode = "ru"):
    if mode == "ru":
        pad_h  = pad_size[0]-img.shape[0]
        pad_w = pad_size[1]-img.shape[1]
        padded_image = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), 'constant', constant_values=0)
        return padded_image
    if mode == "mid":
        pass


def resize_transform(img,label,resize_size):

    return