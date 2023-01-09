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
    |cfg.py(這邊調整bs epoch 等整個專案的參數)
data- 
    |-sup 輔助資料集
    |-test- 測試資料集
          |-test
          |-test.csv
    |-train---train_img
            |-train_GT_keypoints.json
lib- (所有的東西)
    |-core (設計演算法)
        |-acc.py -定義指針角度的計算方法和錶面讀值的轉換
        |-loss.py -設計supervised training和suptrain的loss
    |-dataset (data的處理方法、資料增強)
        |-data_aug.py -資料增強的code
        |-dataset.py -trainingdata,supdata,testdata的dataset處理
        |-preprocessing 資料預處理
    |-model (神經網路)
        |-backbone.py 定義residual block
        |-model.py 定義本專題的網路架構
    |-runner (規劃訓練、測試的流程)
        |-implementation.py -用來測試training sup test data的可視化結果
        |-logger.py -紀錄訓練的loss值 並每五個ep存訓練權重
        |-tester.py -跑測試資料並輸出csv檔
        |-trainer.py -設計supervised training和sup train的訓練流程
    |-util (可視化結果，用來看訓練結果)
        |-visulization -設計可視化的流程並顯示(幫助訓練)
weights- 存放訓練資料
    |-
.gitignore 
main.py

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
