

import cv2
import math
import numpy as np

def meterlike(img,key_points):
    """
    將key points秀在圖片上,然後標誌指針的方向、角度、位置。
    """
    image = img.copy()
    image = visual(image,key_points,visual = False)
    #麻煩死了

    return
def img_show(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
def visual(img,key_points,visual = False):
    print(key_points)
    """
    將key points秀在圖片上
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
    print(p)
    for i in range(4):
        image = cv2.circle(image,p[i],radius=5,color = color[i], thickness=-1)
    if visual:
        img_show(image)

    return image