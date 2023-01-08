
import numpy as np
import math
import numpy as np
def calc_gauge_value(angle,maxangle):
    value = 15*(angle/maxangle)
    if value>15:
        value = 15.0
    return f"{value:.1f}"
def calc_angle(v1, v2):

    r =np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1,2)*np.linalg.norm(v2, 2)))
    deg = r * 180 / np.pi
    a1 = np.array([*v1, 0])
    a2 = np.array([*v2,0])
    a3 = np.cross(a1, a2)
    if np.sign(a3[2]) > 0:
        deg = 360 - deg
    return deg
    
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
    
    #計算指針角度
    angle = calc_angle(
        [
            p[3][0]-p[2][0],
            p[3][1]-p[2][1]
        ],
        [
            p[0][0]-p[2][0],
            p[0][1]-p[2][1]
        ]
        )
    #計算與最大值的角度
    maxangle = calc_angle(
        [
            p[1][0]-p[2][0],
            p[1][1]-p[2][1]
        ],
        [
            p[0][0]-p[2][0],
            p[0][1]-p[2][1]
        ]
        )
    return angle,maxangle
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
    """