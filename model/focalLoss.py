import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, num_classes, box_len, num_anchor, background=1, cuda=1):
        super(FocalLoss, self).__init__()
        self.name = 'FocalLoss'
        self.num_cls = num_classes + background
        self.box_len = box_len
        self.num_anchor = num_anchor
        self.one_hot = torch.eye(num_classes + background).cuda()

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2
        t = self.one_hot[y.data, :]
        t = Variable(t)
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    
    def forward(self, cls_pred, box_pred, cls_truth, box_truth):
        pos = cls_truth > 0 #0 is background
        num_pos = pos.data.long().sum()
        #calculate box location loss
        mask = pos.unsqueeze(2).expand_as(box_pred)      # [batch, anchors, 4]
        masked_box_pred = box_pred[mask].view(-1, self.box_len)      # [#pos,4]
        masked_box_truth = box_truth[mask].view(-1, self.box_len)    # [#pos,4]
        box_loss = F.smooth_l1_loss(masked_box_pred, masked_box_truth, size_average=False)
        #calculate the classify loss
        cls_pred = cls_pred.view(-1, self.num_cls)
        cls_truth = cls_truth.view(-1)
        cls_loss = self.focal_loss(cls_pred, cls_truth)
        #get the loss
        num_pos = num_pos.float()
        #print('cls_loss: %.3f | box_loss: %.3f' % (cls_loss.data[0]/num_pos, box_loss.data[0]/num_pos))
        loss =  (cls_loss + box_loss) / num_pos
        self.cls_loss = cls_loss.data / num_pos
        self.box_loss = box_loss.data / num_pos
        self.loss = loss
        return loss


