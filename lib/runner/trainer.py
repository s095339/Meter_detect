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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class trainer:
    def __init__(self,cfg,trainset,valset = None,arg = None):
        #------
        self.cfg = cfg
        self.arg = arg
        #------
        self.lr = cfg.TRAIN.LR0
        self.bs = cfg.TRAIN.BS
        self.ep = cfg.TRAIN.EPOCH
        self.trainloader = DataLoader(trainset, batch_size=self.bs, shuffle=True)
        self.val = False
        if valset != None:
            self.val = True
            self.valloader = DataLoader(valset, batch_size=1, shuffle=True)
        #initialize----------------------------------
        #model---------------------------------
        self.model = MAJIYABAKUNet(cfg = self.cfg, arg = self.arg).to(device)
        #LOSS-------------------------------------
        if self.cfg.TRAIN.LOSS == "":
            self.loss_fn = nn.MSELoss(reduction="sum")#會改loss
        else:
            self.loss_fn = eval(self.cfg.TRAIN.LOSS)
        #optim--------------------------------------
        if self.cfg.TRAIN.OPTIM.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.cfg.TRAIN.OPTIM.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.cfg.TRAIN.OPTIM.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #--------------------------------------------------------
    def save_model(self):
        import os,time
        result = time.localtime()
        dirname = f"{result.tm_year}{result.tm_mon}{result.tm_mday}\
            _{result.tm_hour}_{result.tm_min}_{self.cfg.MODEL.BACKBONE}"
        os.mkdir()
        os.mkdir(self.cfg.TRAIN.SAVEPTH,dirname)
        savepth = os.path.join(self.cfg.TRAIN.SAVEPTH,dirname)
        torch.save(self.model.state_dict(), savepth)
    def validate(self):
        for _, (X,y) in enumerate(self.valloader):
            pred = self.model(X)
        pass
    def train_loop(self):
        size = len(self.trainloader.dataset)
        for batch, (X, y) in enumerate(self.trainloader):
            # Compute prediction and loss
            X = X.to(device).float()
            y = y.to(device).float()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    def run(self):
        for ep in range(self.ep):
            print(ep)
            self.train_loop()
            if self.val:
                self.validate()
        self.save_model()