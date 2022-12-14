

import torch
import torch.nn as nn
import torch.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def WeightsMse(pred,label):
    """
    想法:錶面的中心點是最critical的,如果中心點的位置錯了，那其他點再怎麼正確
    都是沒用。所以讓中心點的loss比較大
    其次重要的是錶的最大值最小值，最不重要的是指針的位置

    label = [a,b,c,d]
    pred = [a_pred,b_pred,c_pred,d_pred]
    loss = w0||a-a_pred||+w1||b-b_pred||+w2||c-c_pred||+w3||d-d_pred||
    """
    point_weights = [1,1, #最小值
                    1,1,  #最大值
                    1.5,1.5,  #中心點
                    1,1]  #指針值
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


def RaidusLoss(pred):
    """
    loss function for self supervised learning
    預計作法：
    將sup裡面的圖片拿出來做。
    所有的圖片丟下去預測，各自得到一個角度，理論上角度要一樣。第一個圖片是轉正的圖片,
    以他的角度(Theta_base)為基準,取所有預測角度(Theta_x)的mean跟variance
    mean = mean(theta_x)
    variance = variance(theta_x)
    Loss = alpha*(Theta_base-mean)+beta*(variance)

    管他會不會增加acc,反正報告可以寫出很酷的算式
    """
    return