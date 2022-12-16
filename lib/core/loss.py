

import torch
import torch.nn as nn
import torch.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#calculate alg---------------------------------------
#LOSS------------------------------------------------
def WeightsMse(pred,label):
    """
    想法:錶面的中心點是最critical的,如果中心點的位置錯了，那其他點再怎麼正確
    都是沒用。所以讓中心點的loss比較大
    其次重要的是錶的最大值最小值，最不重要的是指針的位置

    label = [a,b,c,d]
    pred = [a_pred,b_pred,c_pred,d_pred]
    loss = w0||a-a_pred||+w1||b-b_pred||+w2||c-c_pred||+w3||d-d_pred||
    """
    point_weights = [1.,1., #最小值
                    1.,1.,  #最大值
                    1.5,1.5,  #中心點
                    1.,1.]  #指針值
    loss_fn = nn.MSELoss()
    weight_stack = []
    for i in range(pred.shape[0]):#bs
        weight_stack.append(point_weights)
    
    weights = torch.tensor(weight_stack).to(device)
    weights = torch.sqrt(weights)

    loss = loss_fn(weights*pred,weights*label)
    #點點：最小值，最大值，中心，指針值。
    #print(loss)
 
    return loss


#Unsupervised Loss------------------------------------------
import math

def dist(p0,p1):
    eps = 0.01
    dist = abs(torch.sqrt(torch.pow(p1[0]-p0[0],2)+torch.pow(p1[1]-p0[1],2)+ eps))
    return dist
def RaidusVarLoss(pred):
    """
    loss function for self supervised learning
    將sup裡面的圖片拿出來做。
    所有的圖片丟下去預測，然後用預測的點來計算角度
    計算指針角度跟最大角度，理論上都要一樣
    然後計算各自的variance
    """
    #print(pred.shape)
    #determind the angle 
    anglelist = [] #指針的角度
    minmaxanglelist = [] #最小值到最大值的總角度
    for key_point in pred:    
        angle = 0.0
        #點點：最小值，最大值，中央值，指針值。
        p = [(0,0),(0,0),(0,0),(0,0)]
        for i in range(4):
            p[i] = (key_point[i*2],key_point[i*2+1])
        #指針的角度計算===============================
        a = dist(p[0],p[3])#a:最小值到指針值的直線距離
        b = dist(p[3],p[2])#b:指針到中心的直線距離
        c = dist(p[2],p[0])#c:中心到最小值的直線距離
        temp = (torch.pow(b,2)+torch.pow(c,2)-torch.pow(a,2))
        temp = temp/(2*b*c+0.001)
        angle = torch.acos(temp).unsqueeze(0)#.unsqueeze(0)
        anglelist.append(angle)
        #最小值到最大值的角度計算=====================
        a = dist(p[0],p[1])#a:最小值到最大值的直線距離
        b = dist(p[1],p[2])#b:最大值到中心的直線距離
        c = dist(p[2],p[0])#c:中心到最小值的直線距離
        temp = (torch.pow(b,2)+torch.pow(c,2)-torch.pow(a,2))
        temp = temp/(2*b*c+0.001)
        angle = torch.acos(temp).unsqueeze(0)#.unsqueeze(0)
        minmaxanglelist.append(angle)

    #rerrange
    batchangle = torch.cat(anglelist,0)
    batchminmaxangle = torch.cat(minmaxanglelist,0)
    #print(batchangle)
    #print(batchminmaxangle)

    #計算角度的variance
    LOSS= torch.var(batchangle)+torch.var(batchminmaxangle)
    #print("LOSS = ",LOSS)
    return LOSS