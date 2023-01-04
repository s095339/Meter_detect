import numpy  as np
import cv2
from PIL import Image
import random
import math
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
"""
1.旋轉、2.上下左右平移
3.加上胡椒點雜訊
4.鏡像翻轉
"""

def rotate(src_img,key_points):
    pivot_point = (int(src_img.shape[0]/2),int(src_img.shape[1]/2))
    shape_img = src_img.shape[0:2]
    #angle:+-pi
    angle_of_rotation = random.random()*2*np.pi-np.pi
    #1.create rotation matrix with numpy array
    rotation_mat = np.transpose(np.array([[np.cos(angle_of_rotation),-np.sin(angle_of_rotation)],
                            [np.sin(angle_of_rotation),np.cos(angle_of_rotation)]]))
    h,w = shape_img
    
    pivot_point_x =  pivot_point[0]
    pivot_point_y = pivot_point[1]
    
    new_img = np.zeros(src_img.shape,dtype='u1') 

    for height in range(h): #h = number of row
        for width in range(w): #w = number of col
            xy_mat = np.array([[width-pivot_point_x],[height-pivot_point_y]])
            
            rotate_mat = np.dot(rotation_mat,xy_mat)

            new_x = pivot_point_x + int(rotate_mat[0])
            new_y = pivot_point_y + int(rotate_mat[1])


            if (0<=new_x<=w-1) and (0<=new_y<=h-1): 
                new_img[new_y,new_x] = src_img[height,width]
    
    p = [(0,0),(0,0),(0,0),(0,0)]
    newlabel = np.array([0,0,0,0,0,0,0,0],dtype=np.float64)
    for i in range(4):
        p[i] = (key_points[i*2].astype(np.uint64),key_points[i*2+1].astype(np.uint64))
    i = 0
    for point in p:
        x = point[0]
        y = point[1]

        xy_mat = np.array([[x-pivot_point_x],[y-pivot_point_y]])
        rotate_mat = np.dot(rotation_mat,xy_mat)
        new_x = pivot_point_x + int(rotate_mat[0])
        new_y = pivot_point_y + int(rotate_mat[1])
        newlabel[i] = new_x
        i=i+1
        #print(i)
        newlabel[i] = new_y
        i=i+1
        

    return new_img,newlabel  
#下面的功能都有夠簡單
def shift(img,label):
    """
    將圖片做上下左右隨機方向隨機大小平移
    平移量大概25~75吧?
    """ 
    
    #隨機產生方向跟位移量
    x_direction = math.pow(-1,round(random.random()))
    y_direction = math.pow(-1,round(random.random()))
    
    choose = [(1,1),(0,1),(1,0)]
    c = choose[int(random.random()*100)%3]
    dx = int((np.random.random()*50+25)*x_direction)*c[0]
    dy = int((np.random.random()*50+25)*y_direction)*c[1]
    
    X = np.roll(img, dy, axis=0)
    X = np.roll(X, dx, axis=1)

    newlabel = [0,0,0,0,0,0,0,0]
    #print(f"dx = {dx},dy = {dy}")
    for i in [0,2,4,6]:
        #print(i)
        newlabel[i] = label[i]+dx
    for i in [1,3,5,7]:
        #print(i)
        newlabel[i] = label[i]+dy
    

    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    return X,newlabel

def donothing(img,label):

    return img,label

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
            min_y_to_top = min_y_to_top
        if Ytobottom < min_y_to_bottom:
            min_y_to_bottom = Ytobottom
    limit_y = min(min_y_to_top,min_y_to_bottom)
    limit_x = min(min_x_to_left,min_x_to_right)
    return limit_x,limit_y

def augshift(image,key_points):
    limit_x,limit_y = cal_limit(image,key_points)
    x_direction = math.pow(-1,round(random.random()))
    y_direction = math.pow(-1,round(random.random()))
    #print(key_points)
    #print(limit_x,limit_y)
    choose = [(1,1),(0,1),(1,0)]
    c = choose[int(random.random()*100)%3]
    dx = int((np.random.random()*limit_x-5)*x_direction)*c[0]
    dy = int((np.random.random()*limit_y-5)*y_direction)*c[1]
    kps = KeypointsOnImage([
            Keypoint(x=key_points[0], y=key_points[1]),
            Keypoint(x=key_points[2], y=key_points[3]),
            Keypoint(x=key_points[4], y=key_points[5]),
            Keypoint(x=key_points[6], y=key_points[7])
        ], shape=image.shape)
    seq = iaa.Sequential([
        iaa.TranslateX(px=(0, dx)),
        iaa.TranslateY(px=(0, dy))
    ])
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    #print(kps_aug[0].x)
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

def imrotate(image,key_points):

    ia.seed(1)
    angle_of_rotation = (random.random()*2-1)*180

    kps = KeypointsOnImage([
            Keypoint(x=key_points[0], y=key_points[1]),
            Keypoint(x=key_points[2], y=key_points[3]),
            Keypoint(x=key_points[4], y=key_points[5]),
            Keypoint(x=key_points[6], y=key_points[7])
        ], shape=image.shape)

    seq = iaa.Sequential([
     # change brightness, doesn't affect keypoints
        iaa.Affine(
            rotate=angle_of_rotation,
            scale=(1, 1)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    image_aug, kps_aug = seq(image=image, keypoints=kps)
    #print(kps_aug[0].x)
    newlabel = [0,0,0,0,0,0,0,0]
    for i in range(0,8,2):
        #print(i,end=",")
        #print(i/2,end=",")
        newlabel[i] = kps_aug[int(i/2)].x
        newlabel[i+1] = kps_aug[int(i/2)].y

# image with keypoints before/after augmentation (shown below)
    #print(image_aug.shape)

    return image_aug,newlabel

def mixaug(image,key_points):
    image,key_points = imrotate(image,key_points)
    image,key_points = augshift(image,key_points)
    return image,key_points