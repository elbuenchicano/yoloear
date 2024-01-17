import  shutil
import  os 
import  yaml

from    tqdm            import tqdm
from    utils           import *
from    torchvision.io  import read_image
from    PIL             import Image
from    ultralytics     import YOLO

import matplotlib.pyplot as plt
import cv2
import numpy             as np

def pad32(im):
    w, h, c = im.shape
    
    rw  = w%32
    rh  = h%32

    nw  = w + (32 - rw) 
    nh  = h + (32 - rh) 

    newim   = np.zeros((nw, nh, 3))

    newim[:w, :h] += im

    #plt.imshow(newim)
    #plt.show()

    return newim.astype(int)

####
def organizeEar():
    '''
    this function organize the folder of ear images
    '''
    
    tsize   = 0.8 
    im_size = 320
    
    
    im_pts  = {
        'win'   : 'I:/db/earvn1/train',
        'linux' : '/home/rensso/Images'
        }

    roots   = {
        'win'   : 'I:/research/yoloear/data',
        'linux' : '/home/rensso/research/yoloear/data'
        }

    im_pt   = im_pts[u_whichOS()]
    data    = roots[u_whichOS()]

    files   = u_listFileAll(im_pt, 'jpg')
    
    nfiles  = len(files)
    bound   = int(nfiles * tsize)

    print('image number ', nfiles)

    train   = files[:bound]
    val     = files[bound:]

    print('train number ', len(train))
    print('val number ', len(val))

    ## building directories....................................................
    
    images  = data + '/' + 'images'
    labels  = data + '/' + 'labels'
        
    i_train = images + '/' + 'train'
    i_val   = images + '/' + 'val'

    l_train = labels + '/' + 'train'
    l_val   = labels + '/' + 'val'

    u_mkdir(data)
    u_mkdir(images)
    u_mkdir(labels)
    u_mkdir(i_train)
    u_mkdir(i_val)
    u_mkdir(l_train)
    u_mkdir(l_val)

    ## copying files...........................................................

    for im in tqdm(train, 'train'):
               
        name    = os.path.basename(im[:-4])

        im      = cv2.imread(im)
        #im      = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        w, h, c = im.shape 

        if w > im_size or h > im_size:
            nim  = cv2.resize(im, (im_size, im_size), interpolation = cv2.INTER_AREA)
            w, h = im_size, im_size


        else:
            nim     = np.zeros((im_size, im_size, 3), dtype= 'int')
            nim[:w, :h] = im 



        dims = np.array([w/2, h/2, w, h])

        dims = dims/im_size
        
        im_ = i_train + '/' + name + '.jpg' 
        lb_ = l_train + '/' + name + '.txt'  

        row = f'0 {dims[0]} {dims[1]} {dims[2]} {dims[3]}' 
                
        f = open(lb_, "w")
        f.write(row)
        f.close()
            
        cv2.imwrite(im_, nim)

        
    for im in tqdm(val, 'val'):
               
        name = os.path.basename(im[:-4])

        im   = cv2.imread(im)

        nim  = np.zeros((im_size, im_size, 3), dtype= 'int')

        w, h, c = im.shape 

        if w > im_size or h > im_size:
            nim  = cv2.resize(im, (im_size, im_size), interpolation = cv2.INTER_AREA)
            w, h = im_size, im_size


        else:
            nim     = np.zeros((im_size, im_size, 3), dtype= 'int')
            nim[:w, :h] = im 

        dims = np.array([w/2, h/2, w, h])

        dims = dims/im_size

        im_ = i_val + '/' + name + '.jpg' 
        lb_ = l_val + '/' + name + '.txt'  

        row = f'0 {dims[0]} {dims[1]} {dims[2]} {dims[3]}' 
                
        f = open(lb_, "w")
        f.write(row)
        f.close()
            
        cv2.imwrite(im_, nim)
    
    ######## yaml file dump ...................................................

    info = {
        'path'  : data,
        'train' : 'images/train',
        'val'   : 'images/val',

        'names': {
            0: 'ear'
            }
        }
        
    file=open("data.yaml","w")
    yaml.dump(info, file)
    file.close()


def yolotrain():
    model   = YOLO('yolov8n.pt')
    results = model.train(data='data.yaml', epochs=5)

##############################################################
def show():

    img     = "I:/db/download_2023-12-27_13-50-43/mov_001_007585.jpeg"
    txt1    = "I:/db/download_2023-12-27_13-50-43/mov_001_007585.txt"

    img     = "I:/db/download_2023-12-27_13-50-43/mov_001_007586.jpeg"
    txt1    = "I:/db/download_2023-12-27_13-50-43/mov_001_007586.txt"

    img = read_image(img)
    img = np.transpose(img, (1,2,0))

    #print(img.shape)
    #plt.imshow(img)
    #plt.show()

    img_w, img_h = img.shape[1], img.shape[0]
    
    with open(txt1) as f:
        lines_txt = f.readlines()
        lines = []
        for line in lines_txt:
            lines.append([int(line.split()[0])] + [round(float(el), 5) for el in line.split()[1:]])

    bboxes = []
    

    # In this loop we convert normalized coordinates to absolute coordinates
    for line in lines:
        # Number 0 is a class of rectangles related to bounding boxes.
        x_c, y_c, w, h = round(line[1] * img_w), round(line[2] * img_h), round(line[3] * img_w), round(line[4] * img_h)
        bboxes.append([round(x_c - w/2), round(y_c - h/2), round(x_c + w/2), round(y_c + h/2)])

    img = img.numpy()

    for bbox_idx, bbox in enumerate(bboxes):
        top_left_corner, bottom_right_corner = tuple([bbox[0], bbox[1]]), tuple([bbox[2], bbox[3]])
        img = cv2.rectangle(img, bottom_right_corner, top_left_corner, (0,255,0), 3)
    
    
        
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.show()
                



##############################################################
if __name__ == '__main__':
    #show()
    organizeEar()
    #yolotrain()
    ...
