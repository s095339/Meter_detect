import cv2
import numpy as np

from lib.dataset.data_aug import rotate,shift,noise,mirror_flip
from lib.util.visualization import visual

def change_angle_to_radius_unit(angle):
    angle_radius = angle * (np.pi/180)
    return angle_radius

if __name__ == '__main__':
    label = np.array([
            148.78973388671875,
            256.58038330078125,
            237.25320434570312,
            244.67828369140625,
            183.66751098632812,
            182.5,
            121.33609008789062,
            139.2275390625
        ])
    img = cv2.imread("./data/train/train_img/scale_52_meas_0.png")
    
    from lib.dataset.preprocessing import resize,label_fit

    resized_img = resize(img,[640,640])
    newlabel = label_fit(img,resized_img,label)
    print(newlabel)
    visual(resized_img,newlabel,isvisual = True)
    
    #visual(resized_img,newlabel,isvisual = True)
    height,width = resized_img.shape[:2] #get width and height of image
    angle = 90
    img_,label_ =  rotate(resized_img.copy(),newlabel)
    print(label_)
    visual(img_,label_,isvisual = True)
    """

    img_,label_  = shift(resized_img.copy(),newlabel)
    print(label_)
    visual(img_,label_,isvisual = True)
    """