import argparse 
import os

import torch

from config.cfg import _C as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Do anything you want through command line')
    parser.add_argument('--mode',
                        help='train or test',
                        type=str,
                        default=''
                        )

    args = parser.parse_args()

    return args
def test(arg,cfg):
    pass
def train(arg,cfg):
    
    #prepare data-----------------
    from lib.dataset.dataset import MeterDataset
    
    from torch.utils.data import random_split
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    target_transform = torch.tensor
    dataset = MeterDataset(cfg = cfg,
                            transform = transform,
                            target_transform = target_transform
                            )
    trainset = dataset
    #trainset,valset = random_split(dataset, [8000,2000])
    #show img
    from lib.util.visualization import visual
    image,label  = next(iter(trainset))
    img = image.detach().numpy().transpose(1,2,0)
    label = label.detach().numpy()
    #點點：最小值，最大值，中央值，指針值。
    #visual(img,label,visual = True)

    from lib.core.acc import angle_calculate
    print(angle_calculate(label,mode = "degree"))

    
    #----------------------------------
    from lib.runner.trainer import trainer
    Trainer = trainer(cfg = cfg,
                      trainset = trainset,
                      arg = arg)

    Trainer.run()
    
def main(arg,cfg):
    if arg.mode == "train":
        train(arg,cfg)
    elif arg.mode == 'test':
        test(arg.cfg)
    else:
        pass


    return

if __name__ == '__main__':
    args = parse_args()
    main(args,cfg)
