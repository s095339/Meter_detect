import matplotlib.pyplot as plt
import logging
import datetime
import os
import numpy as np
class logger:
    def __init__(self,cfg,mode = "train"):
        self.cfg = cfg
        self.mode = mode
        if mode =="train":
            #紀錄訓練參數(做報告)------
            self.lr = cfg.TRAIN.LR0
            self.bs = cfg.TRAIN.BS
            self.ep = cfg.TRAIN.EPOCH
            self.losstype = cfg.TRAIN.LOSS
            #------------------------
        if mode.lower() =="suptrain":
            self.lr = cfg.SUPTRAIN.LR0
            self.bs = "None"
            self.ep = cfg.SUPTRAIN.EPOCH
            self.losstype = cfg.SUPTRAIN.LOSS
        #------------------------
        self.log_root = cfg.LOG.DIR
        #logging basic setup-----------
        self.log_setup()
        #------------------------------
        self.loss_record = []
    def log_setup(self):
        """
        創建Log資料夾(如果沒有的話)
        然後創建訓練資料夾來記錄訓練的參數和loss的趨勢。
        """
        if not os.path.exists(self.log_root):
            os.mkdir("log")
        #make training dir to record training status
        dir_name = f"{self.mode}_"+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.dirpath = os.path.join(self.log_root,dir_name)
        os.mkdir(self.dirpath)
        
        self.log_filename = os.path.join(self.dirpath,"train_log.log")
        logging.basicConfig(level=logging.INFO, filename=self.log_filename, filemode='w',
        #format='[%(levelname).1s %(asctime)s] %(message)s',
        format='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %H:%M:%S',
        )
        self.logger = logging.getLogger("trainlogger")
        if self.mode =="train":
            self.logger.warning(f"[Supervised learning] epoch = {self.ep},batch = {self.bs},loss = {self.losstype},lr = {self.lr}")
        if self.mode.lower() =="suptrain":
            self.logger.warning(f"[Unsupervised Supportdata learning] epoch = {self.ep},batch = {self.bs},loss = {self.losstype},lr = {self.lr}")
            
    def log_insert(self,ep,batch,loss):
        """
        訓練過程中記錄他的loss變化。並寫在log檔內。
        """
        #print(f"epoch = {ep},batch = {batch},loss = {loss:>7f}")
        self.logger.info(f"epoch = {ep},batch = {batch},loss = {loss:>7f}")
        if len(self.loss_record)<ep+1:#ep = 1 2 3 4 5....
            print("----",ep)
            self.loss_record.append([])
        self.loss_record[ep].append(loss)
    def export_loss_plot(self):
        """
        訓練完成之後,輸出loss log來觀察下降趨勢。
        """
        loss_mean = np.mean(np.array(self.loss_record),axis = 0)
        xaxis = [i+1 for i in range(len(loss_mean))]
        plt.plot(xaxis,loss_mean)
        plt.title("loss_log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(os.path.join(self.dirpath,"losslog.png"))
        plt.show()
        
        


        

