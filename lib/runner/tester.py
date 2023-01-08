import torch
from torch.utils.data import DataLoader
from lib.model.model import MAJIYABAKUNet

#for validate----
from lib.core.acc import angle_calculate,calc_gauge_value
#------
from tqdm import tqdm
import os,time
import pandas as pd
#------



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class tester:
    def __init__(self,cfg,testset,arg = None):
        #------
        self.cfg = cfg
        self.arg = arg
        #supervised learning------
        testset.get_imgname = True
        self.testloader = DataLoader(testset, batch_size=1, shuffle=False)
        
        #load model---------------------------------
        self.model = MAJIYABAKUNet(cfg = self.cfg, arg = self.arg).to(device)
        self.savepth = cfg.TEST.OUTPUTPATH
        #pretrined weights--------------------------------
        if self.cfg.PRETRAIN:
            print(f"loading pretrained weight:{self.cfg.PRETRAIN}")
            weights = torch.load(self.cfg.PRETRAIN)
            self.model.load_state_dict(weights)
        #acc-------------------------------------
        self.angle_calculate = angle_calculate
        #csv
        self.csv = pd.read_csv(cfg.TEST.CSVFILE)
    def csv_save(self):
        result = time.localtime()
        name = f"test_{result.tm_mday}_{result.tm_hour}_{result.tm_min}.csv"
        path = os.path.join(self.savepth,name)
        self.csv.to_csv(path)
        print(f"save csvfile to {self.savepth} as {name}")
    def validate(self):
        progress = tqdm(total = len(self.testloader))
        self.model.eval()
        #X = next(iter(self.loader))
        for batch,(X,name) in enumerate(self.testloader):    
            name = name[0] #提取tuple value
            #print("file:",name)
            X = X.to(device).float()
            with torch.no_grad():
                pred = self.model(X.clone())
            pred  = pred.cpu().detach().numpy().squeeze()
            #計算表面的角度
            angle,maxangle = self.angle_calculate(pred,mode = "degree")
            #計算表面讀值
            value = calc_gauge_value(angle,maxangle)
            #將讀值寫進去csv
            index = self.csv[self.csv['name']==name].index.values[0]
            self.csv['label'][index] = value
            progress.update(1)
        self.csv_save()

    def run(self):
        self.validate()
        
