

import cv2
import math
import numpy as np

def img_show(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
def dist(p0,p1):
    dist = abs(math.sqrt(math.pow(p1[0]-p0[0],2)+math.pow(p1[1]-p0[1],2)))
    return dist
def visual(img,key_points,isvisual = False):
    """
    將key points貼在圖片上並輸出
    """
    image = img.copy()
    p = [(0,0),(0,0),(0,0),(0,0)]
    color = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,255)
    ]
    #記錄各個點
    #點點：最小值，最大值，中央值，指針值。
    for i in range(4):
        p[i] = (key_points[i*2].astype(np.uint64),key_points[i*2+1].astype(np.uint64))
    #把點點畫上去
    #print(p)
    for i in range(4):
        try:
            image = cv2.circle(image,p[i],radius=5,color = color[i], thickness=-1)
        except:
            print(p)
    if isvisual:
        img_show(image)

    return image

def meterlike(img,key_points,isvisual = False):
    """
    將key points秀在圖片上,然後標誌指針的方向、角度、位置。
    """
    image = img.copy()
    p = [(0,0),(0,0),(0,0),(0,0)]
    color = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,255)
    ]

    for i in range(4):
        p[i] = (key_points[i*2].astype(np.uint64),key_points[i*2+1].astype(np.uint64))
    #把點點畫上去
    image = img.copy()
    image = visual(image,key_points)

    image = cv2.arrowedLine(image,p[2],p[3],(0,0,255),thickness = 3)
    image = cv2.line(image,p[2],p[0],(0,255,0),thickness = 2)
    #麻煩死了
    if isvisual:
        img_show(image)
    return

def ShowGrayImgFromTensor(img,label):
    img = img.cpu().detach().squeeze().numpy()
    label = label.cpu().detach().squeeze().numpy()
    visual(img,label,isvisual = True)

