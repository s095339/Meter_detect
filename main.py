import argparse 
import os
from lib.dataset.dataset import testDataset,MeterDataset,SupportDatset
from torch.utils.data import random_split
from torchvision import transforms
import torch

from config.cfg import _C as cfg

#---
from lib.util.visualization import img_show

def parse_args():
    parser = argparse.ArgumentParser(description='Do anything you want through command line')
    parser.add_argument('--mode',
                        help='train or test',
                        type=str,
                        default=''
                        )
    parser.add_argument('--impleset',
                        help='dataset',
                        type=str,
                        default=''
                        )
    args = parser.parse_args()

    return args
def test(arg,cfg):
    pass
def train_sup(arg,cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset =  SupportDatset(
        cfg = cfg,
        transform = transform
    )
    """
    data = next(iter(dataset))
    print(data.shape)
    img = data[3].detach().numpy().transpose(1,2,0)
    img_show(img)
     """
    from lib.runner.trainer import sup_trainer
    trainer = sup_trainer(
        cfg = cfg,
        trainset = dataset,
        arg = arg
    )
    trainer.run()
def implement(arg,cfg):
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    """
    transform = transforms.Compose([
                    transforms.ToTensor(), 
                    #transforms.Normalize([0.5], [0.5])
                    ])
    """
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                 ])
    """
    invTrans = transforms.Compose([ 
                                    transforms.Normalize(mean = [ -0.5, -0.5],
                                                     std = [ 1., 1.]),
                                 ])
    target_transform = torch.tensor
    if arg.impleset == "test" or "":
        dataset = testDataset(cfg = cfg,
                                transform = transform,
                                )
    elif arg.impleset == "train":
        dataset = MeterDataset(cfg = cfg,
                                transform = transform,
                                target_transform=torch.tensor
                                )
    from lib.runner.implementation import implementer
    Implementer = implementer(cfg = cfg,
                      dataset = dataset,
                     # inv_train = invTrans,
                      arg = arg)
    Implementer.run(test_number=10)
def train(arg,cfg):
    #prepare data-----------------
    from lib.dataset.dataset import MeterDataset
    
    from torch.utils.data import random_split
    from torchvision import transforms

    transform = transforms.Compose([
                    transforms.ToTensor(), 
                    #transforms.Normalize([0.5], [0.5])
                    ])
    target_transform = torch.tensor
    dataset = MeterDataset(cfg = cfg,
                            transform = transform,
                            target_transform = target_transform
                            )
    trainset = dataset
    #trainset,valset = random_split(dataset, [9000,1000])
    #show img
    """
    from lib.util.visualization import visual,meterlike
    image,label  = trainset[9999]
    img = image.detach().numpy().transpose(1,2,0)
    label = label.detach().numpy()
    #點點：最小值，最大值，中央值，指針值。
    print(label)
    print(img.shape)
    visual(img,label,isvisual = True)
    #meterlike(img,label,isvisual = True)
    

    from lib.core.acc import angle_calculate
    print(angle_calculate(label,mode = "degree"))

    """
    #----------------------------------
    from lib.runner.trainer import trainer
    Trainer = trainer(cfg = cfg,
                      trainset = trainset,
                      #valset = valset,
                      arg = arg)
    Trainer.run()
    
def main(arg,cfg):
    if arg.mode == "train":
        train(arg,cfg)
    elif arg.mode == 'test':
        test(arg,cfg)
    elif arg.mode == 'train_sup' or arg.mode == 'suptrain':
        train_sup(arg,cfg)
    else: # arg.mode == "implement":
        implement(arg,cfg)
    return
if __name__ == '__main__':
    args = parse_args()
    main(args,cfg)
