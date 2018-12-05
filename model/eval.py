import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class evalRetina(nn.Module):
    def __init__(self, width, height, nms_thresh=0.5, obj_thresh=0.4):
        self.width = width
        self.height = height
        self.nms_thresh = nms_thresh
        self.obj_thresh = obj_thresh
        self.anchors = torch.Tensor([
            [0.0239749, 0.0904281],
            [0.2569985, 0.3775594],
            [0.0125396, 0.0429755],
            [0.0255381, 0.0382968],
            [0.0090712, 0.0187574],
            [0.0456727, 0.0533909],
            [0.1276380, 0.1695816],
            [0.0189022, 0.0215612],
            [0.0669595, 0.1037171]
        ]).cuda()
        self.names = ["background","bus","traffic light","traffic sign","person","bike","truck","motor","car","train","rider"]
        self.class_num = 11
        self.anchor_num = 9
        self.W1 = width // 32
        self.H1 = height // 32
        self.W2 = width // 16
        self.H2 = height // 16
        self.W3 = width //8
        self.H3 = height //8
        self.make_anchors()
        
    def forward(self, cls_preds, box_preds, cls_truths=None, box_truths=None):
        B, _, _ = cls_preds.size()
        result_list = []
        for i in range(B):
            cls_pred = cls_preds[i,:,:]
            box_pred = box_preds[i,:,:]
            score, labels = cls_pred.sigmoid().max(1)
            ids1 = labels > 0
            ids2 = score > self.obj_thresh
            ids = ids1 * ids2
            box_pred = self.correct_box(box_pred)
            if ids.sum() == 0:
                result = None
            else:
                obj_prob = score[ids].contiguous().cpu()
                obj_cls = labels[ids].contiguous().cpu()
                obj_box = box_pred[ids].contiguous().cpu()
                prob, cls_, box = self.box_nms(obj_prob, obj_cls, obj_box)
                prob = prob.view(-1, 1)
                cls_ = cls_.view(-1, 1).float() - 1
                box = box.view(-1, 4)
                result = torch.cat((prob, box, cls_), 1)
            result_list.append(result)
            if cls_truths is not None:
                cls_truth = cls_truths[i,:]
                box_truth = box_truths[i,:,:]
                box_truth = self.correct_box(box_truth)
                ids = cls_truth > 0
        return result_list

    def correct_box(self, box):
        box = box.exp()
        box_xy = box[:, 0:2]
        box_wh = box[:, 2:4]
        box_xy = box_xy - 0.000001 + self.anchor_xywh[:, 0:2]
        box_xy1 = box_xy[0:self.len1,:]
        box_xy1[:, 0] = box_xy1[:, 0] / self.W1
        box_xy1[:, 1] = box_xy1[:, 1] / self.H1
        box_xy2 = box_xy[self.len1:(self.len1 + self.len2), :]
        box_xy2[:, 0] = box_xy2[:, 0] / self.W2
        box_xy2[:, 1] = box_xy2[:, 1] / self.H2
        box_xy3 = box_xy[(self.len1 + self.len2):, :]
        box_xy3[:, 0] = box_xy3[:, 0] / self.W3
        box_xy3[:, 1] = box_xy3[:, 1] / self.H3
        box_xy = torch.cat((box_xy1, box_xy2, box_xy3), 0)
        box_wh = box_wh * self.anchor_xywh[:, 2:]
        box = torch.cat((box_xy, box_wh), 1)
        return box

    def box_nms(self, obj_prob, obj_cls, obj_box):
        num_box, _ = obj_box.size()
        obj_prob, order = obj_prob.sort(0, descending=True)
        obj_cls = obj_cls[order]
        obj_box = obj_box[order, :]
        area = obj_box[:, 2] * obj_box[:, 3]
        wt = torch.Tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]])
        obj_box = obj_box.mm(wt) # transfor xywh to x1y1x2y2 order
        mask = torch.ones(num_box).byte()
        for i in range(num_box-1):
            if mask[i] > 0:
                iou = self.iou_box_nms(obj_box[i, :].view(1,4), obj_box[(i+1):, :].view(-1,4))
                mask_cls = (obj_cls[(i+1):] == obj_cls[i])
                mask_box = iou > self.nms_thresh
                need_mask = mask[(i+1):]
                need_mask[mask_cls * mask_box] = 0
                mask[(i+1):] = need_mask
        return obj_prob[mask], obj_cls[mask], obj_box[mask]

    def iou_box_nms(self, box1, box2):
        #box1 is [1, 4], box2 is [N, 4] 
        N, _ = box2.size()
        zero = torch.zeros(N)
        box = box1.view(1,4).expand(N, 4)
        left = torch.max(box[:, 0], box2[:, 0])
        up = torch.max(box[:, 1], box2[:, 1])   
        right = torch.min(box[:, 2], box2[:, 2])
        down = torch.min(box[:, 3], box2[:, 3]) 
        intersection_w = torch.max( right.sub(left), zero)
        intersection_h = torch.max( down.sub(up), zero)
        intersection = intersection_w * intersection_h
        area1 = (box1[0,2] - box1[0,0]) * (box1[0,3] - box1[0,1])
        area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
        union = area1 + area2 - intersection
        iou = intersection / union
        return iou

    def make_anchors(self):
        #x = (exp(tx) - 0.000001 + w)/W
        #y = (exp(ty) - 0.000001 + h)/H
        #w = exp(tw) / anchor[idx][0]
        #h = exp(th) / anchor[idx][1]
        self.len1 = self.W1 * self.H1 * self.anchor_num
        self.len2 = self.W2 * self.H2 * self.anchor_num
        self.len3 = self.W3 * self.H3 * self.anchor_num
        len_all = (self.len1 + self.len2 + self.len3)//self.anchor_num
        anchor_wh = self.anchors.view(1,self.anchor_num,2).expand(len_all, self.anchor_num,2).contiguous()
        a = np.arange(self.W1)
        b = np.arange(self.H1)
        x, y = np.meshgrid(a, b)
        x = torch.from_numpy(x).view(self.H1, self.W1, 1).cuda()
        y = torch.from_numpy(y).view(self.H1, self.W1, 1).cuda()
        xy1 = torch.cat((x,y),2).view(self.H1*self.W1, 1, 2).expand(self.H1*self.W1, self.anchor_num, 2).contiguous()
        a = np.arange(self.W2)
        b = np.arange(self.H2)
        x, y = np.meshgrid(a, b)
        x = torch.from_numpy(x).view(self.H2, self.W2, 1).cuda()
        y = torch.from_numpy(y).view(self.H2, self.W2, 1).cuda()
        xy2 = torch.cat((x,y),2).view(self.H2*self.W2, 1, 2).expand(self.H2*self.W2, self.anchor_num, 2).contiguous()
        a = np.arange(self.W3)
        b = np.arange(self.H3)
        x, y = np.meshgrid(a, b)
        x = torch.from_numpy(x).view(self.H3, self.W3, 1).cuda()
        y = torch.from_numpy(y).view(self.H3, self.W3, 1).cuda()
        xy3 = torch.cat((x,y),2).view(self.H3*self.W3, 1, 2).expand(self.H3*self.W3, self.anchor_num, 2).contiguous()
        anchor_xy = torch.cat((xy1, xy2, xy3), 0).float()
        self.anchor_xywh = torch.cat((anchor_xy, anchor_wh), 2).contiguous().view(-1, 4)