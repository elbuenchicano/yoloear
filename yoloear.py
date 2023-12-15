from PIL import Image
from ultralytics import YOLO

import shutil
import os 
import yaml

from tqdm   import tqdm
from utils  import *


####
def organizeEar():
    '''
    this function organize the folder of ear images
    '''
    im_pts = {
        'win'   : '"I:/db/earvn1/Images"',
        'linux' : '/home/rensso/Images'
        }

    roots = {
        'win'   : 'I:/research/yoloear',
        'linux' : '/home/rensso/research/yoloear'
        }

    im_pt       = im_pts[u_whichOS()]

    item_list   = u_listFileAllDic()

##############################################################
if __name__ == '__main__':
    ...
