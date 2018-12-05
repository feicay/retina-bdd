import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T 
import visdom
from torch.utils import data
from torch.autograd import Variable
import time
import argparse
import numpy as np
import os
from PIL import Image
from model.retinanet import RetinaNet
from model.eval import evalRetina
import cv2
import math

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Testing')
parser.add_argument('--pth', default='./checkpoint/retina-bdd-backup.pth', type=str, help='load model from checkpoint')
parser.add_argument('--img', default='./002.jpg', type=str, help='test image')
parser.add_argument('--vis', default=1, type=int, help='visdom')
parser.add_argument('--thresh', default=0.5, type=float, help='visdom')
args = parser.parse_args()

def plot_boxes_cv2(image, boxes, class_names=None, color=None, fps=None):
    if boxes is None:
        return image
    img = image
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)
    width = img.shape[1]
    height = img.shape[0]
    num, _ = boxes.size()
    for i in range(num):
        box = boxes[i, :]
        x1 = int(box[1] * width)
        y1 = int(box[2] * height)
        x2 = int(box[3] * width)
        y2 = int(box[4] * height)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[0]
            cls_id = int(box[5])
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            str_prob = '%.2f'%cls_conf
            info = class_names[cls_id] + str_prob
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, info, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if fps is None:
        savename = 'prediction.png'
        print("save plot results to %s" %savename)
        cv2.imwrite(savename, img)
    else:
        fps_info = 'fps:' + '%.2f'%fps
        img = cv2.putText(img, fps_info, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 1)
    return img

def detect_image(image, network, thresh, names):
    evaluator = evalRetina(416, 416, obj_thresh=0.4)
    pil_img = Image.open(image)
    w_im, h_im = pil_img.size
    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img = pil_img.resize( (416, 416) )
    img = transform(img).cuda()
    img = img.view(1,3,416,416)
    t0 = time.time()
    cls_pred, box_pred = network(img)
    t1 = time.time()
    result_list = evaluator.forward(cls_pred, box_pred)
    t2 = time.time()
    print(result_list[0])
    print('inference time : %f, nms time : %f'%((t1-t0), (t2-t1)))
    image1 = cv2.imread(image)
    im = plot_boxes_cv2(image1, result_list[0], names)
    cv2.imshow('prediction',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

def test():
    print('initializing network...')
    network = RetinaNet(3, 10, 9)
    checkpoint = torch.load(args.pth)
    network.load_state_dict(checkpoint['net'])
    network = network.cuda().eval()
    class_names = ["bus","traffic light","traffic sign","person","bike","truck","motor","car","train","rider"]

    image = args.img
    img_tail =  image.split('.')[-1] 
    if img_tail == 'jpg' or img_tail =='jpeg' or img_tail == 'png':
        detect_image(image, network, args.thresh, class_names)   
    elif img_tail == 'mp4' or img_tail =='mkv' or img_tail == 'avi' or img_tail =='0':
        detect_vedio(image, network, args.thresh, class_names)
    else:
        print('unknow image type!!!')
    
test()