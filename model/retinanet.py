import torch
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

class Conv2dBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride=1, bn=1, activation='relu'):
        super(Conv2dBlock, self).__init__()
        self.name = 'Conv2dBlock'
        layers = []
        pad = (kernel_size - stride + 1) // 2
        if bn:
            layers.append( nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride, padding=pad, bias=False) )
            layers.append( nn.BatchNorm2d(outchannel) )
        else:
            layers.append( nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride, padding=pad) )
        if activation == 'relu':
            layers.append( nn.ReLU(inplace=True) )
        self.module = nn.Sequential(*layers)
        init.kaiming_uniform(self.module[0].weight.data, mode='fan_in')
    def forward(self, x):
        out = self.module(x)
        return out

class UpsampleAddBlock(nn.Module):
    def __init__(self, inchannel_left, outchannel):
        super(UpsampleAddBlock, self).__init__()
        self.name = 'UpsampleAddBlock'
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel_left, outchannel, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        init.kaiming_uniform(self.left[0].weight.data, mode='fan_in')
    def forward(self, x_left, x_up):
        x1 = self.left(x_left)
        x2 = self.up(x_up)
        return F.relu(x1 + x2)

class HeaderTopBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(HeaderTopBlock, self).__init__()
        self.name = 'HeaderTopBlock'
        channel = inchannel // 2
        self.module = nn.Sequential(
            Conv2dBlock(inchannel, channel, 1),
            Conv2dBlock(channel, inchannel, 3),
            Conv2dBlock(inchannel, channel, 1),
            Conv2dBlock(channel, inchannel, 3),
            Conv2dBlock(inchannel, outchannel, 1)
        )
    def forward(self, x):
        out = self.module(x)
        return out

class HeaderBottomBlock(nn.Module):
    def __init__(self, inchannel, clschannel):
        super(HeaderBottomBlock, self).__init__()
        self.name = 'HeaderBottomBlock'
        self.module = nn.Sequential(
            Conv2dBlock(inchannel, 256, 3),
            Conv2dBlock(256, clschannel, 1, bn=0, activation='linear')
        )
    def forward(self, x):
        x = self.module(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualLayer, self).__init__()
        self.name = 'ResidualLayer'
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, inchannel//2, 1, bias=False),
            nn.BatchNorm2d(inchannel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel//2, outchannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        init.kaiming_uniform(self.left[0].weight.data, mode='fan_in')
        init.kaiming_uniform(self.left[3].weight.data, mode='fan_in')
    def forward(self, x):
        y = self.left(x) + x
        return F.relu(y)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, block_num):
        super(ResidualBlock, self).__init__()
        self.name = 'ResidualBlock'
        self.layer1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride=2, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        layers = []
        for i in range(block_num):
            layers.append(ResidualLayer(outchannel, outchannel))
        self.layer2 = nn.Sequential(*layers)
        init.kaiming_uniform(self.layer1[0].weight.data, mode='fan_in')
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y) 
        return y

class Darknet53(nn.Module):
    def __init__(self, inchannel):
        super(Darknet53, self).__init__()
        self.name = 'Darknet53'
        self.block1 = Conv2dBlock(inchannel, 32, 3)
        self.block2 = ResidualBlock(32, 64, 1)
        self.block3 = ResidualBlock(64, 128, 2)
        self.block4 = ResidualBlock(128, 256, 8)
        self.block5 = ResidualBlock(256, 512, 8)
        self.block6 = ResidualBlock(512, 1024, 4)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x1 = self.block4(x)
        x2 = self.block5(x1)
        x3 = self.block6(x2)
        return x1, x2, x3

#use the darknet53 backbone network instead of resnet101 to speed up
class RetinaNet(nn.Module):
    def __init__(self, inchannel, num_class, num_anchors):
        super(RetinaNet, self).__init__()
        self.cls_num = num_class + 1
        cls_channel = (num_class + 1) * num_anchors
        box_channel = 4 * num_anchors
        self.backbone = Darknet53(inchannel)
        self.header1 = HeaderTopBlock(1024, 512)   
        self.cls1 = HeaderBottomBlock(512, cls_channel)
        self.box1 = HeaderBottomBlock(512, box_channel)
        self.up2x = UpsampleAddBlock(512, 512)
        self.header2 = HeaderTopBlock(512, 256)
        self.cls2 = HeaderBottomBlock(256, cls_channel)
        self.box2 = HeaderBottomBlock(256, box_channel)
        self.up4x = UpsampleAddBlock(256, 256)
        self.header3 = HeaderTopBlock(256, 128)
        self.cls3 = HeaderBottomBlock(128, cls_channel)
        self.box3 = HeaderBottomBlock(128, box_channel)
    def forward(self, x):
        #the output has three scales, 1/32, 1/16, 1/8
        #the cls output size: Batch * [(W*H/32/32 + W*H/16/16 + W*H/8/8)*anchors] * classes
        #the box output size: Batch * [(W*H/32/32 + W*H/16/16 + W*H/8/8)*anchors] * 4
        x1, x2, x3 = self.backbone(x)
        y1 = self.header1(x3)
        cls1 = self.cls1(y1)
        box1 = self.box1(y1)
        y2 = self.up2x(x2, y1)
        y2 = self.header2(y2)
        cls2 = self.cls2(y2)
        box2 = self.box2(y2)
        y3 = self.up4x(x1, y2)
        y3 = self.header3(y3)
        cls3 = self.cls3(y3)
        box3 = self.box3(y3)
        B, _, _, _ = cls1.size()
        cls_pred1 = cls1.permute(0,2,3,1).contiguous().view(B, -1, self.cls_num)
        box_pred1 = box1.permute(0,2,3,1).contiguous().view(B, -1, 4)
        cls_pred2 = cls2.permute(0,2,3,1).contiguous().view(B, -1, self.cls_num)
        box_pred2 = box2.permute(0,2,3,1).contiguous().view(B, -1, 4)
        cls_pred3 = cls3.permute(0,2,3,1).contiguous().view(B, -1, self.cls_num)
        box_pred3 = box3.permute(0,2,3,1).contiguous().view(B, -1, 4)
        cls_pred = torch.cat((cls_pred1, cls_pred2, cls_pred3), 1)
        box_pred = torch.cat((box_pred1, box_pred2, box_pred3), 1)
        #return cls1, box1, cls2, box2, cls3, box3
        return cls_pred, box_pred

def test():
    net = RetinaNet(3, 5, 9)
    x = torch.randn(1,3,64,64)
    x = Variable(x)
    cls1, box1, cls2, box2, cls3, box3 = net(x)
    print(cls1.size())
    print(box1.size())
    print(cls2.size())
    print(box2.size())
    print(cls3.size())
    print(box3.size())

#test()