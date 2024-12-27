# -*- coding:utf-8 -*-
# @Author: Luo Ge
# @Email: geyouguang@163.com
# @File : train.py
# @Time : 2024/11/20

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import loader
from torch.utils.data import DataLoader
import cv2
from CDMIF import CDMIF


# *-----------------*
# hyper-parameters
imgshape = (500, 500)
srcadir = '/root/autodl-tmp/dataset/multi-exposure/trainset1/SourceA'
srcbdir = '/root/autodl-tmp/dataset/multi-exposure/trainset1/SourceB'
gtdir = '/root/autodl-tmp/dataset/multi-exposure/trainset1/Label'
# *-----------------*


class Trainer:
    def __init__(self):
        self.epoch = 40
        self.batch_size = 15
        self.lr = 0.0005

        print("=====> Loading datasets")
        self.train_set = loader(srcadir, srcbdir, gtdir)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        print("=====> Building model")
        self.model = CDMIF()
        self.model = self.model.cuda()
        self.criterion = nn.MSELoss(reduction='mean')

        print("=====> Setting Optimizer")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40)
        self.train_loss = []

        if os.path.exists('model/latest.pth'):
            print('=====> Loading pre-trained model...')
            state = torch.load('model/latest.pth')
            self.train_loss = state['train_loss']
            print(len(self.train_loss))
            self.model.load_state_dict(state['model'])

    def train(self):
        seed = 42
        print("=====> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for epoch in range(self.epoch):
            epoch_loss = []
            for batch, (srcapath, srcbpath, gtpath) in enumerate(self.train_loader):
                Gt, X1, X2 = [], [], []
                for i in range(len(srcapath)):
                    img_Gt = cv2.imread(gtpath[i])
                    Gt.append(np.expand_dims(cv2.resize(img_Gt, imgshape, interpolation=cv2.INTER_LANCZOS4), 0))
                    img_X1 = cv2.imread(srcapath[i])
                    X1.append(np.expand_dims(cv2.resize(img_X1, imgshape, interpolation=cv2.INTER_LANCZOS4), 0))
                    img_X2 = cv2.imread(srcbpath[i])
                    X2.append(np.expand_dims(cv2.resize(img_X2, imgshape, interpolation=cv2.INTER_LANCZOS4), 0))
                Gt = torch.from_numpy(np.concatenate(Gt, axis=0)).float().cuda()
                X1 = torch.from_numpy(np.concatenate(X1, axis=0)).float().cuda()
                X2 = torch.from_numpy(np.concatenate(X2, axis=0)).float().cuda()
                self.optimizer.zero_grad()
                F = self.model(X1, X2)
                loss = self.criterion(F, Gt)
                loss = loss * 1000
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                if batch % 10 == 0:
                    print('Epoch:{} \t Cur/All:{}/{} \t Avg Loss:{:.4f}'.format(epoch, batch+1, len(self.train_loader), loss.item()/1000))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss)/1000)
            print(np.mean(epoch_loss)/1000)

            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            os.makedirs("model/", exist_ok=True)
            torch.save(state, os.path.join('model/', 'latest.pth'))
            torch.save(state, os.path.join('model/', str(epoch) + '.pth'))

        print('=====> Finished Training!')


if __name__ == '__main__':
    instance = Trainer()
    instance.train()
