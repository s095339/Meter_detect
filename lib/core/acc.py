
import numpy as np
import math

def dist(p0,p1):
    dist = abs(math.sqrt(math.pow(p1[0]-p0[0],2)+math.pow(p1[1]-p0[1],2)))
    return dist
def angle_calculate(key_point,mode = "radians"):
    """
    ### 計算錶面的角度值
    paramter
    ------------------
    key_point:輸入label或錶面的預測值
    mode:預設為"radians",代表return [0,pi],輸入"degree"則return 角度值

    return:
    ------------------
    錶面角度
    """
    angle = 0.0
    p = [(0,0),(0,0),(0,0),(0,0)]
    for i in range(4):
        p[i] = (key_point[i*2].astype(np.float64),key_point[i*2+1].astype(np.float64))
 
    #print(p)
    #點點：最小值，最大值，中央值，指針值。
    a = dist(p[0],p[3])#a:最小值到指針值的直線距離
    b = dist(p[3],p[2])#b:指針到中心的直線距離
    c = dist(p[2],p[0])#c:中心到最小值的直線距離
    temp = (math.pow(b,2)+math.pow(c,2)-math.pow(a,2))
    temp = temp/(2*b*c)
    angle = math.acos(temp)
    



    if mode.lower() == "degree":
        angle = np.rad2deg(angle)
    
    return angle