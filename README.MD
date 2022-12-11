# 1 Meter Detect

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
用之前做車道辨識的環境就OK了

## 3.train
For training, copy the command:
>python main.py --mode train

##