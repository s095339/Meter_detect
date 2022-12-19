import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,1)     


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.BACKBONE = 'ResNet34_GRAY'  # resnet隨便你選



# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = './data/train/train_img'       # the path of images folder
_C.DATASET.LABELROOT = './data/train/train_GT_keypoints.json'      # the path of det_annotations folder
_C.DATASET.PREPROCESS = 'resize' #resize #看是要padding  勸你不要用padding
_C.DATASET.PADDINGSIZE = [720 , 720]
_C.DATASET.IMGSIZE = [640,640]
_C.DATASET.GRAYSCALE = True
#pretrain
#test 或training的時候的pretrain weight
_C.PRETRAIN = "" #"./weights/model_ep50_bs12 _normal.pth"#"./weights/model_ep50_bs12 _normal.pth
# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.BS = 4
_C.TRAIN.EPOCH = 2
_C.TRAIN.OPTIM = "adam" #或 SGD 或 Adagrad
_C.TRAIN.LOSS = "WeightsMse"
_C.TRAIN.SAVEPTH = "./weights" #訓練好的權重存在這邊'

#self-supervised
_C.SUPTRAIN = CN(new_allowed=True)
_C.SUPTRAIN.LR0 = 0.0001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.SUPTRAIN.BS = 8 #必須是48的因數。
_C.SUPTRAIN.EPOCH = 2
_C.SUPTRAIN.OPTIM = "adam" #或 SGD 或 Adagrad
_C.SUPTRAIN.LOSS = "RaidusVarLoss"
_C.SUPTRAIN.SAVEPTH = "./weights" #訓練好的權重存在這邊'
_C.SUPTRAIN.DATAROOT = "./data/sup"

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.DATAROOT = "./data/test/test"
_C.TEST.PREPROCESS = "resize"
_C.TEST.PADDINGSIZE = [720 , 720]
_C.TEST.IMGSIZE = [640,640]
#loggin
_C.LOG = CN(new_allowed=True)
_C.LOG.DIR = "./log"




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