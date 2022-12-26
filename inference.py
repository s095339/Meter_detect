import cv2
import numpy as np

from lib.dataset.data_aug import rotate,shift,imrotate
from lib.util.visualization import visual

def change_angle_to_radius_unit(angle):
    angle_radius = angle * (np.pi/180)
    return angle_radius

if __name__ == '__main__':
    label_ = np.array([
            148.78973388671875,
            256.58038330078125,
            237.25320434570312,
            244.67828369140625,
            183.66751098632812,
            182.5,
            121.33609008789062,
            139.2275390625
        ])
    im_ = cv2.imread("./data/train/train_img/scale_52_meas_0.png")
    
    #im,label = imrotate(img,label)
    from lib.dataset.preprocessing import augresize
    from lib.dataset.data_aug import KeepSizeResize,augshift

     
    #im,label = imrotate(img,label)
    #from lib.dataset.preprocessing import resize,label_fit
    #resized_img = resize(im,[640,640])
    #newlabel = label_fit(im,resized_img,label)
    visual(im_,label_,isvisual = True)
    for i in range(30):
        #im,label = augshift(im_,label_)
        
        im,label = augresize(im_,[480,480],label_)
        im,label = KeepSizeResize(im,label)
        im,label = imrotate(im,label)
        visual(im,label,isvisual = True)
        print(im.shape)
    #visual(im,label,isvisual = True)
    """
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
    

    img_,label_  = shift(resized_img.copy(),newlabel)
    print(label_)
    visual(img_,label_,isvisual = True)
    """