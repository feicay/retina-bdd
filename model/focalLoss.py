import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, num_classes, box_len, num_anchor):
        super(FocalLoss, self).__init__()
        self.name = 'FocalLoss'
        self.num_cls = num_classes
        self.box_len = box_len
        self.num_anchor = num_anchor
        self.one_hot = torch.eye(num_classes + 1)

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        t = self.one_hot[y.data.cpu(), :]
        t = Variable(t).cuda() 
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

