import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,1)     


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.BACKBONE = 'resnet34'  # 34 被我鎖住了



# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = './data/train/train_img'       # the path of images folder
_C.DATASET.LABELROOT = './data/train/train_GT_keypoints.json'      # the path of det_annotations folder
_C.DATASET.PREPROCESS = 'augresize' #resize #看是要padding  勸你不要用padding
_C.DATASET.PADDINGSIZE = [720 , 720]
_C.DATASET.IMGSIZE = [480,480]
_C.DATASET.GRAYSCALE = True

_C.DATAAUG = CN(new_allowed=True)
_C.DATAAUG.ENABLE = True
_C.DATAAUG.TYPE = ["imrotate","augshift","mixaug"]
_C.DATAAUG.DATARATIO = 1 #0.1~1.0 這個數字代表著一次的ep裡面幾成的data要做aug
_C.DATAAUG.AUGRATIO = [0,0,8] #一成的旋轉，四成的
#pretrain
#test 或training的時候的pretrain weight
_C.PRETRAIN = "./weights/SUPTRAIN_202314_22_18_resnet34/SUPTRAIN_model_ep5_bs8.pth"#"./weights/model_ep100_bs8.pth"
# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.BS = 16
_C.TRAIN.EPOCH = 50
_C.TRAIN.OPTIM = "adam" #或 SGD 或 Adagrad
_C.TRAIN.LOSS = "WeightsMse"
_C.TRAIN.POINT_WEIGHTS = [1.2,1.2, #最小值
                    1.,1.,  #最大值
                    1.5,1.5,  #中心點
                    1.1,1.1]#for WeightsMse

_C.TRAIN.SAVEPTH = "./weights" #訓練好的權重存在這邊'

#self-supervised
_C.SUPTRAIN = CN(new_allowed=True)
_C.SUPTRAIN.ENABLE = True
_C.SUPTRAIN.CYCLE = 3#每train幾次跑一次sup資料
_C.SUPTRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.SUPTRAIN.BS = 8 #必須是48的因數。
_C.SUPTRAIN.EPOCH = 5
_C.SUPTRAIN.OPTIM = "adam" #或 SGD 或 Adagrad
_C.SUPTRAIN.LOSS = "RaidusVarLoss"
_C.SUPTRAIN.SAVEPTH = "./weights" #訓練好的權重存在這邊'
_C.SUPTRAIN.DATAROOT = "./data/sup"

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.DATAROOT = "./data/test/test"
_C.TEST.PREPROCESS = "augresize"
_C.TEST.PADDINGSIZE = [720 , 720]
_C.TEST.IMGSIZE = [480,480]
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