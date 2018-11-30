import os
import random
import json
import math
from PIL import Image, ImageEnhance
from PIL import ImageFile
import torch
import numpy as np
import torchvision as tv 
from torch.utils import data
from torchvision import transforms as T 

imagedir = '/home/adas/data/bdd100k/images/100k'
labeldir = '/home/adas/data/bdd100k/labels_raw/100k'
#the kmeans result as anchor box init size
anchors = np.array([
    [0.0239749, 0.0904281],
    [0.2569985, 0.3775594],
    [0.0125396, 0.0429755],
    [0.0255381, 0.0382968],
    [0.0090712, 0.0187574],
    [0.0456727, 0.0533909],
    [0.1276380, 0.1695816],
    [0.0189022, 0.0215612],
    [0.0669595, 0.1037171]
],dtype=np.float32)
bdd_names = ["bus","traffic light","traffic sign","person","bike","truck","motor","car","train","rider"]

class bddDataset(data.Dataset):
    def __init__(self, width, height, truth=1, data_expand=1, train=1):
        self.names = ["background","bus","traffic light","traffic sign","person","bike","truck","motor","car","train","rider"]
        self.class_num = 11
        self.anchors = torch.from_numpy(anchors)
        self.anchor_num = 9
        self.width = width
        self.height = height
        self.imageList = []
        self.labelList = []
        if train:
            dir_img = imagedir + '/train'
            dir_lab = labeldir + '/train'
        else:
            dir_img = imagedir + '/val'
            dir_lab = labeldir + '/val'
        image_file_list = os.listdir(dir_img)
        for f in image_file_list:
            self.imageList.append(dir_img + '/' + f)
            label_file_name = dir_lab + '/' + f.replace('.jpg', '.json')
            objs_truth = self.get_objs(label_file_name)
            self.labelList.append(objs_truth)
        self.len = len(self.imageList)
        self.train = train
        self.truth = truth
        self.data_expand = data_expand
        if data_expand:
            self.saturation = 1.5
            self.exposure = 1.5
            self.hue = 1.5
            self.sharpness = 1.5
        self.transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        self.seen = 0
        #print(self.labelList[9999])
        #cls_truth, box_truth = self.make_label(self.labelList[9999], 416, 416)
        #pos = cls_truth > 0
        #print(cls_truth[pos], box_truth[pos, :] )
        #print(self.imageList[100], self.labelList[100])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        imgDir = self.imageList[index]
        pil_img = Image.open(imgDir)
        mode = pil_img.mode
        #ignore the gray images
        while mode != 'RGB':
            index = int(random.random()*self.len)
            imgDir = self.imageList[index]
            pil_img = Image.open(imgDir)
            mode = pil_img.mode
        if self.train:
            #random the input size from (10~16)*32
            if (self.seen)%1280 == 1 and self.seen > 6400:
                self.width_now = (random.randint(0,6) + 10)*32
                self.height_now = (random.randint(0,6) + 10)*32
                print('resizing input %d x %d'%(self.width_now,self.height_now))
            img = pil_img.resize( (self.width_now, self.height_now) )
            if self.data_expand:
                #Brightness,Color,Contrast,Sharpness range from 0.5~1.5
                #change exposure
                enh_bri = ImageEnhance.Brightness(img)
                brightness = self.exposure - random.random() 
                img = enh_bri.enhance(brightness)
                #change color
                enh_col = ImageEnhance.Color(img)
                color = self.saturation - random.random()
                img = enh_col.enhance(color)
                #change Contrast 
                enh_con = ImageEnhance.Contrast(img)
                Contrast = self.hue - random.random()
                img = enh_con.enhance(Contrast)
                #change Sharpness
                enh_sha = ImageEnhance.Sharpness(img)
                sharpness =  self.sharpness - random.random()
                img = enh_sha.enhance(sharpness)
        else:
            img = pil_img.resize( (self.width, self.height) )
        image = self.transform(img)
        if self.truth:
            cls_truth, box_truth = self.make_label(self.labelList[index])

    def get_objs(self, label_file):
        with open(label_file,'r') as load_f:
            load_dict = json.load(load_f)
            num = len(load_dict['frames'][0]['objects'])
            objs = []
            for i in range(num):
                obj_name = load_dict['frames'][0]['objects'][i]["category"]
                if str(obj_name) in bdd_names:
                    x1 = float(load_dict['frames'][0]['objects'][i]["box2d"]["x1"]) / 1280
                    x2 = float(load_dict['frames'][0]['objects'][i]["box2d"]["x2"]) / 1280
                    y1 = float(load_dict['frames'][0]['objects'][i]["box2d"]["y1"]) / 720
                    y2 = float(load_dict['frames'][0]['objects'][i]["box2d"]["y2"]) / 720
                    cls_ = bdd_names.index(obj_name) + 1
                    obj = torch.Tensor([cls_,x1,y1,x2,y2]).view(1,5)
                    objs.append(obj)
            obj_truth = torch.cat(objs, 0)
        return obj_truth

    def iou_anchor(self, box_truth, anchors):
        #box_truth: [x1,y1,x2,y2]    anchor:[[w1,h1],[w2,h2]...]
        #this function will find the best shape of anchor box for the object
        num_anchor, _ = anchors.size()
        box_truth = box_truth.view(1, 4).expand(num_anchor, 4)
        W_truth = box_truth[:,2] - box_truth[:,0]
        H_truth = box_truth[:,3] - box_truth[:,1]
        min_w = torch.min(W_truth, anchors[:,0])
        min_h = torch.min(H_truth, anchors[:,1])
        intersection = min_w * min_h
        union = W_truth * H_truth + anchors[:,0] * anchors[:,1] - intersection
        iou = intersection / union
        max_iou, max_idx = iou.max(0)
        return max_iou, max_idx

    def box_xyxy2xywh(self, box):
        box_out = box.clone()
        box_out[0] = (box[2] - box[0])/2 + box[0]
        box_out[1] = (box[3] - box[1])/2 + box[1]
        box_out[2] = box[2] - box[0]
        box_out[3] = box[3] - box[1]
        return box_out

    def make_anchors(self, scale):
        w = int(self.width_now * scale + 0.5)
        h = int(self.height_now * scale + 0.5)
        i = np.arange(w)
        j = np.arange(h)
        x, y = np.meshgrid(i, j)
        X = torch.from_numpy(x) + 0.5
        Y = torch.from_numpy(y) + 0.5
        anchor_grid = torch.zeros()
    
    def make_label(self, obj_truth, width, height):
        #the output has three scales, 1/32, 1/16, 1/8
        #the cls label size: Batch * [(W*H/32/32 + W*H/16/16 + W*H/8/8)*anchors] * 1
        #the box label size: Batch * [(W*H/32/32 + W*H/16/16 + W*H/8/8)*anchors] * 4
        obj_num, _ = obj_truth.size()
        W1 = width // 32
        H1 = height // 32
        W2 = width // 16
        H2 = height // 16
        W3 = width // 8
        H3 = height // 8
        cls_truth1 = torch.zeros(W1*H1, self.anchor_num)
        cls_truth2 = torch.zeros(W2*H2, self.anchor_num)
        cls_truth3 = torch.zeros(W3*H3, self.anchor_num)
        box_truth1 = torch.zeros(W1*H1, self.anchor_num, 4)
        box_truth2 = torch.zeros(W2*H2, self.anchor_num, 4)
        box_truth3 = torch.zeros(W3*H3, self.anchor_num, 4)
        for i in range(obj_num):
            obj_cls = obj_truth[i, 0].clone()
            obj_box = obj_truth[i, 1:5].clone()
            _, idx_anchor = self.iou_anchor(obj_box, self.anchors)
            box_xywh = self.box_xyxy2xywh(obj_box)
            w1 = int(W1 * box_xywh[0])
            h1 = int(H1 * box_xywh[1])
            tx1 = math.log(W1 * box_xywh[0] - w1 + 0.000001)
            ty1 = math.log(H1 * box_xywh[1] - h1 + 0.000001)
            tw = math.log(box_xywh[2] / self.anchors[idx_anchor][0])
            th = math.log(box_xywh[3] / self.anchors[idx_anchor][1])
            cls_truth1[(h1*W1 + w1), idx_anchor] = obj_cls
            box_truth1[(h1*W1 + w1), idx_anchor, :] = torch.Tensor([tx1, ty1, tw, th])
            w2 = int(W2 * box_xywh[0])
            h2 = int(H2 * box_xywh[1])
            tx2 = math.log(W2 * box_xywh[0] - w2 + 0.000001)
            ty2 = math.log(H2 * box_xywh[1] - h2 + 0.000001)
            cls_truth2[(h2*W2 + w2), idx_anchor] = obj_cls
            box_truth2[(h2*W2 + w2), idx_anchor, :] = torch.Tensor([tx2, ty2, tw, th])
            w3 = int(W3 * box_xywh[0])
            h3 = int(H3 * box_xywh[1])
            tx3 = math.log(W3 * box_xywh[0] - w3 + 0.000001)
            ty3 = math.log(H3 * box_xywh[1] - h3 + 0.000001)
            cls_truth3[(h3*W3 + w3), idx_anchor] = obj_cls
            box_truth3[(h3*W3 + w3), idx_anchor, :] = torch.Tensor([tx3, ty3, tw, th])
        cls_truth = torch.cat((cls_truth1, cls_truth2, cls_truth3), 0).view(-1).long()
        box_truth = torch.cat((box_truth1, box_truth2, box_truth3), 0).view(-1, 4)
        return cls_truth, box_truth


def test():
    bdd_trainset = bddDataset(416,416,train=0)
    anchors[:,0] = anchors[:,0]*416
    anchors[:,1] = anchors[:,1]*416
    print(anchors)

test()