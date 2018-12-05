import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import visdom
from torch.utils import data
from torch.autograd import Variable
import time
import argparse
import numpy as np
import os
from PIL import Image
from model.retinanet import RetinaNet
from model.focalLoss import FocalLoss
import data.bdd as bdd

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--vis', default=1, type=int, help='visdom')
parser.add_argument('--ngpus', default=4, type=int, help='number of gpus')
args = parser.parse_args()

def train():
    max_epoch = 110
    lr = 0.001
    step_epoch = 50
    lr_decay = 0.1
    train_batch_size = 32
    val_batch_size = 8
    if args.vis:
        vis = visdom.Visdom(env=u'test1')
    #dataset 
    print('importing dataset...')
    trainset = bdd.bddDataset(416, 416)
    loader_train = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=1, num_workers=4, drop_last=True)
    valset = bdd.bddDataset(416, 416, train=0)
    loader_val = data.DataLoader(valset, batch_size=val_batch_size, shuffle=1, num_workers=4, drop_last=True)
    #model
    print('initializing network...')
    network = RetinaNet(3, 10, 9)
    if args.resume:
        print('Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/retina-bdd-backup.pth')
        network.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    if args.ngpus > 1:
        net = torch.nn.DataParallel(network).cuda()
    else:
        net = network.cuda()
    #criterion
    criterion = FocalLoss(10, 4, 9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    #start training
    for i in range(start_epoch, max_epoch):
        print('--------start training epoch %d --------'%i)
        trainset.seen = 0
        valset.seen = 0
        loss_train = 0.0
        net.train()
        t0 = time.time()
        for ii, (image, cls_truth, box_truth) in enumerate(loader_train):
            image = Variable(image).cuda()
            cls_truth = Variable(cls_truth).cuda()
            box_truth = Variable(box_truth).cuda()
            #forward
            optimizer.zero_grad()
            cls_pred, box_pred = net(image)
            #loss
            loss = criterion(cls_pred, box_pred, cls_truth, box_truth)
            #backward
            loss.backward()
            #update
            optimizer.step()
            loss_train += loss.data
            #print('forward time: %f, loss time: %f, backward time: %f, update time: %f'%((t1-t0),(t2-t1),(t3-t2),(t4-t3)))
            print('%3d/%3d => loss: %f, cls_loss: %f, box_loss: %f'%(ii,i,criterion.loss, criterion.cls_loss, criterion.box_loss))
            if args.vis:
                vis.line(Y=loss.data.cpu().view(1,1).numpy(),X=np.array([ii]),win='loss',update='append' if ii>0 else None)
        t1 = time.time()
        print('---one training epoch time: %fs---'%((t1-t0)))
        if i < 3:
            loss_train = loss.data
        else:
            loss_train = loss_train / ii
        loss_val = 0.0
        net.eval()
        for jj, (image, cls_truth, box_truth) in enumerate(loader_val):
            image = Variable(image).cuda()
            cls_truth = Variable(cls_truth).cuda()
            box_truth = Variable(box_truth).cuda()
            optimizer.zero_grad()
            cls_pred, box_pred = net(image)
            loss = criterion(cls_pred, box_pred, cls_truth, box_truth)
            loss_val += loss.data
            print('val: %3d/%3d => loss: %f, cls_loss: %f, box_loss: %f'%(jj,i,criterion.loss, criterion.cls_loss, criterion.box_loss))
        loss_val = loss_val / jj
        if args.vis:
            vis.line(Y=torch.cat((loss_val.view(1,1), loss_train.view(1,1)),1).cpu().numpy(),X=np.array([i]),\
                        win='eval-train loss',update='append' if i>0 else None)
        print('Saving weights...')
        state = {
            'net': net.module.state_dict(),
            'loss': loss_val,
            'epoch': i,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if (i+5)%d == 0:
            torch.save(state, './checkpoint/retina-bdd-%03d.pth'%i)
        torch.save(state, './checkpoint/retina-bdd-backup.pth')
        if (i+1) % step_epoch == 0:
            lr = lr*0.1
            print('learning rate: %f'%lr)
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    torch.save(network,'retina-bdd_final.pkl')
    print('finished training!!!')


train()