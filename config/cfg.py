import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,1)     


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.BACKBONE = 'resnet34'  # resnet隨便你選



# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = './data/train/train_img'       # the path of images folder
_C.DATASET.LABELROOT = './data/train/train_GT_keypoints.json'      # the path of det_annotations folder
_C.DATASET.PREPROCESS = 'resize' #resize #看是要padding 還是做resize  padding 或 resize
_C.DATASET.PADDINGSIZE = [720 , 720]
_C.DATASET.IMGSIZE = [640,640]

#pretrain
#test 或training的時候的pretrain weight
_C.PRETRAIN = "" #"./weights/20221212_14_16_resnet34/model_ep1_bs4.pth"
# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.BS = 4
_C.TRAIN.EPOCH = 1
_C.TRAIN.OPTIM = "adam" #或 SGD 或 Adagrad
_C.TRAIN.LOSS = "WeightsMse"
_C.TRAIN.SAVEPTH = "./weights" #訓練好的權重存在這邊'
# testing
_C.TEST = CN(new_allowed=True)


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()