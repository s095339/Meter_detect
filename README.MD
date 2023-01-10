# 類比錶面預測Final Project


## 1.環境與前置作業
python=3.7
pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=10.2 
在外面創建data資料夾，然後在裡面創建sup train test三個資料夾
將cropimg sup test的壓縮檔分別解包到train sup test裡面

## 2.file 
上述結束完後，這份專案就應當有以下的檔案
```python=
|config-
    |cfg.py(這邊調整bs epoch 等整個專案的參數)
|data- 
    |-sup 輔助資料集
    |-test- 測試資料集
          |-test
          |-test.csv
    |-train--
            |-train_img
            |-train_GT_keypoints.json
|lib- (所有的東西)
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
|weights- 存放訓練資料
    |-
.gitignore 
main.py

```
## 3.train
去./cfg/cfg.py 的_C.TRAIN內調整bs epoch lr等參數
然後下指令
>python main.py --mode train

log裡面會存放每個ep的權重
這時候外面的weights資料夾會在訓練完之後，會生成一個資料夾檔名為訓練完的時間，裡面有一個weights，
到./cfg/cfg.py 的_C.PRETRAIN後面將weights的路徑改為那個weights之後，將_C.DATASET.NORMALIZE = True改為False。然後就可以跑suptrain
## 4.suptrain
sup資料的訓練
去./cfg/cfg.py 的_C.SUPTRAIN內調整bs epoch lr等參數
>python main.py --mode suptrain

同樣的weights會被存在weights資料夾內，這時候的weights就是最終的weights，
把./cfg/cfg.py 的_C.PRETRAIN改為這個weights。
## 5.test
test
>python main.py --mode test

這時候程式就會自動在./data/test/資料夾底下生成一個新的csv檔案
## 6.測試可視化結果

去config/cfg.py底下的 _C.PRETRAIN 將weights改成想測試的weights
測試在訓練資料的結果
>python main.py --impleset train

測試在測試資料的結果
>python main.py --impleset test

測試在輔助資料的結果
>python main.py --impleset suptrain
#

