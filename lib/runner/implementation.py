import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from lib.model.model import MAJIYABAKUNet
#for loss------
from lib.core.loss import WeightsMse
#for validate----
from lib.core.acc import angle_calculate
#------
from tqdm import tqdm
#------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class implementer:
    """
    隨機對幾筆資料(可以是測試或訓練資料)做implementation並提供可視化結果。
    判斷這個東西到底有沒有用。
    """
    def __init__(self,cfg,dataset,inv_train = None,arg = None):
        #------
        self.cfg = cfg
        self.arg = arg
        #------
        self.loader = DataLoader(dataset, batch_size = 1, shuffle=True)
        self.model = MAJIYABAKUNet(cfg = self.cfg, arg = self.arg).to(device)
        #------
        self.inv_transform = inv_train
        #pretrined weights--------------------------------
        if self.cfg.PRETRAIN:
            print(f"loading pretrained weight:{self.cfg.PRETRAIN}")
            weights = torch.load(self.cfg.PRETRAIN)
            self.model.load_state_dict(weights)
        else:
            raise NotImplementedError("needs pretrained!!!")

    def validate(self):
        self.model.eval()
        X = next(iter(self.loader))
        try:
            X,_ = next(iter(self.loader))
            X = X.to(device).float()
            
        except:
            X = next(iter(self.loader)).to(device).float()
        with torch.no_grad():
            pred = self.model(X.clone())
        from lib.util.visualization import meterlike
        if self.inv_transform:
            X = self.inv_transform(X)
        X  = X.cpu().detach().numpy().squeeze().transpose(1,2,0)
        print(X.shape)
        pred  = pred.cpu().detach().numpy().squeeze()
        meterlike(X,pred,isvisual = True)
        #mean_loss = total_loss/len(self.valloader)
        #print(f"{bcolors.OKGREEN}validate mean loss = :{bcolors.WARNING}{mean_loss}{bcolors.ENDC}")
    def run(self,test_number = 10):
        for i in range(test_number):
            self.validate()