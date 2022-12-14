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
from .logger import logger
#======
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#爽拉我就是要加這個鬼
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
        #logger
        self.logger = logger(cfg)
        #initialize----------------------------------
        #load model---------------------------------
        self.model = MAJIYABAKUNet(cfg = self.cfg, arg = self.arg).to(device)
        print(self.model)
        #pretrined weights--------------------------------
        if self.cfg.PRETRAIN:
            print(f"loading pretrained weight:{self.cfg.PRETRAIN}")
            weights = torch.load(self.cfg.PRETRAIN)
            self.model.load_state_dict(weights)
        #LOSS-------------------------------------
        if self.cfg.TRAIN.LOSS == "":
            self.loss_fn = nn.MSELoss(reduction="sum")#會改loss
        else:
            print("LOSS = ",self.cfg.TRAIN.LOSS)
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
        dirname = f"{result.tm_year}{result.tm_mon}{result.tm_mday}_{result.tm_hour}_{result.tm_min}_{self.cfg.MODEL.BACKBONE}"
        
        dirpath = os.path.join(self.cfg.TRAIN.SAVEPTH,dirname)
        os.mkdir(dirpath)
        savepth = os.path.join(dirpath,f"model_ep{self.ep}_bs{self.bs}.pth")
        torch.save(self.model.state_dict(), savepth)
    def validate(self):
        total_loss = 0.0
        progress = tqdm(total = len(self.valloader))
        self.model.eval()
        for _, (X,y) in enumerate(self.valloader):
            X = X.to(device).float()
            y = y.to(device).float()
            with torch.no_grad():
                pred = self.model(X)
                total_loss += self.loss_fn(pred, y)
                progress.update(1)
        mean_loss = total_loss/len(self.valloader)
        print(f"{bcolors.OKGREEN}validate mean loss = :{bcolors.WARNING}{mean_loss}{bcolors.ENDC}")
    def train_loop(self,ep):
        size = len(self.trainloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.trainloader):
            # Compute prediction and loss
            X = X.to(device).float()
            y = y.to(device).float()
            
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            #loss.backward()
        

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.logger.log_insert(ep = ep,batch = batch,loss = loss)
    def run(self):
        torch.autograd.set_detect_anomaly(True)
        for ep in range(self.ep):
            print("epoch = ",ep)
            self.train_loop(ep)
            if self.val:
                self.validate()
        self.logger.export_loss_plot()
        self.save_model()