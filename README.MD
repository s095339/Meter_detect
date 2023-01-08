# 1 Meter Detect
```python-
TODO list：
1.Add validation 
2.loss function 
3.prediction visualization
4.determinded accuracy 
5.add avalible backbone
6.log file and plot to record training status
```
## training data
label：點點：最小值，最大值，中央值，指針值。

## 1.file 

The repo contains following files and dirs
```python=
config-
    |cfg.py(這邊調整bs epoch 等等)
data-
    |-sup
    |-test
    |-train---train_img
            |-train_GT_keypoints.json
lib-
    |-core (可能要放loss跟acc的算法)
    |-dataset (data的處理方法、資料增強)
    |-model (神經網路)
    |-runner (train,test,(TODO)self-supervised learning)
    |-util(visualization)
weights-
```
## 2.enviroment
1.創建虛擬環境
>conda create --name meter python=3.7
2.下載套件
>conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=10.2 -c pytorch


## 3.train
For training, copy the command:
>python main.py --mode train
log裡面會存放每個ep的權重
## 4.suptrain
sup資料的訓練
>python main.py --mode suptrain

## 5.test
test
>python main.py --mode test
## 6.測試可視化結果

去config/cfg.py底下的 _C.PRETRAIN 將weights改成想測試的weights
測試在訓練資料的結果
>python main.py --impleset train
測試在測試資料的結果
>python main.py --impleset test
測試在輔助資料的結果
>python main.py --impleset suptrain
## 角度
