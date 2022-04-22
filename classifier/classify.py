import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from .src.config import Config
from .src.classifier import Classifier

def get_model(mode=None,attribuite='hair'):
    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    if attribuite=='hair':
        model = Classifier(config,'hair_classification')
    elif attribuite=='gender':
        model = Classifier(config,'gender_classification')
    elif attribuite=='smile':
        model = Classifier(config,'smile_classification')
    model.load()
    return  model



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """





    config_path = os.path.join(os.path.dirname(__file__),'./config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)



    return config

def check_hair(img,model):
    output= model.process(img)
    return output[0][1]>0.09


def check_gender(img,model):
    output= model.process(img)
    return output[0][1]>0.15




