import argparse 
import os
from lib.dataset.dataset import testDataset,MeterDataset,SupportDatset
from torch.utils.data import random_split
from torchvision import transforms
import torch
from lib.runner.trainer import trainer,sup_trainer
from config.cfg import _C as cfg

#---
from lib.util.visualization import img_show
if cfg.DATASET.NORMALIZE:
    transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])
                    ])
    invTrans = transforms.Compose([ 
                                transforms.Normalize(mean = [ -0.5],
                                                    std = [ 1/0.5]),
                                ])
else:
    transform = transforms.Compose([
                    transforms.ToTensor()
                    ])
    invTrans = None
print("transoform:",transform)
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

    dataset =  SupportDatset(
        cfg = cfg,
        transform = transform
    )
    
    from lib.runner.trainer import sup_trainer
    trainer = sup_trainer(
        cfg = cfg,
        trainset = dataset,
        arg = arg
    )
    trainer.run()

    

def implement(arg,cfg):
    
    
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
    else:
        dataset = SupportDatset(cfg = cfg,
                                transform = transform)
    if arg.impleset == "test" or arg.impleset == "train":                          
        from lib.runner.implementation import implementer
        Implementer = implementer(cfg = cfg,
                        dataset = dataset,
                        inv_train = invTrans,
                        arg = arg)
        Implementer.run(test_number=100)
    else:
        from lib.runner.implementation import sup_implementer
        Implementer = sup_implementer(cfg = cfg,
                        dataset = dataset,
                        inv_train = invTrans,
                        arg = arg)
        Implementer.run()




def train(arg,cfg):
    #prepare data-----------------
    target_transform = torch.tensor
    dataset = MeterDataset(cfg = cfg,
                            transform = transform,
                            target_transform = target_transform
                            )
    trainset = dataset
    

    supset =  SupportDatset(
        cfg = cfg,
        transform = transform
    )

    #----------------------------------
    from lib.runner.trainer import trainer
    Trainer = trainer(cfg = cfg,
                      trainset = trainset,
                      supset = supset,
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
    elif arg.mode == 'train_sup2' or arg.mode == "suptrain2":
        train_sup2(arg,cfg)
    else: # arg.mode == "implement":
        implement(arg,cfg)
    return
if __name__ == '__main__':
    args = parse_args()
    main(args,cfg)
