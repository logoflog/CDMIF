# -*- coding:utf-8 -*-
# @Author: Luo Ge
# @Email: geyouguang@163.com
# @File : dataloader.py
# @Time : 2024/11/18

import torch.utils.data as data
import os
import numpy as np
import cv2

gtdir = r'F:\dataset\ir-vis\RoadScene\vis'
srcadir = r'F:\dataset\ir-vis\RoadScene\vis'
srcbdir = r'F:\dataset\ir-vis\RoadScene\ir'

# 只加载图片路径，减少内存消耗，如果直接加载图片，那么在后续的批次划分中会很耗时
class loader(data.Dataset):
    def __init__(self, srcadir, srcbdir, gtdir=None):
        super(loader, self).__init__()
        a = os.listdir(srcadir)
        b = os.listdir(srcbdir)
        if(gtdir is not None):
            self.gtpaths = [gtdir + '/' + x for x in os.listdir(gtdir)]
        self.srcapaths = [srcadir + '/' + x for x in a]
        self.srcbpaths = [srcbdir + '/' + x for x in b]

        # for i in range(len(os.listdir(srcadir))):
        #     if(a[i][:-6]!=b[i][:-6]): print('False '+a[i]+' '+b[i])

        # self.gt = np.load('testset/multi_exposure/test_label.npy', allow_pickle=True)  # (batch,height,width,c)
        # self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        # self.gt_t = torch.from_numpy(self.gt)

        # self.rgb = np.load('testset/multi_exposure/test_under.npy', allow_pickle=True)  # (batch,height,width,c)
        # self.rgb = np.transpose(self.rgb, (0, 3, 1, 2))
        # self.rgb_t = torch.from_numpy(self.rgb)

    def __getitem__(self, item):
        srcapath = self.srcapaths[item]
        srcbpath = self.srcbpaths[item]
        gtpath = self.gtpaths[item]

        return (srcapath, srcbpath, gtpath)

    def __len__(self):
        return len(self.srcapaths)

if __name__ == '__main__':
    dataset = loader(srcadir, srcbdir, gtdir)
    dataloader = data.DataLoader(dataset, batch_size=10)
    for e in range(5):
        for b, (srcapath, srcbpath, gtpath) in enumerate(dataloader):
            X1, X2 = [], []
            for i in range(len(srcapath)):
                X1.append(np.expand_dims(cv2.imread(srcapath[i]), 0))
                X2.append(np.expand_dims(cv2.imread(srcbpath[i]), 0))
                print(srcapath[i], srcbpath[i], X1[-1].shape, X2[-1].shape)
            # X1 = np.concatenate(X1, axis=0)
            # X2 = np.concatenate(X2, axis=0)
            # print(X1.shape)
            # print(X2.shape)
            # print(b)
            # print(gtpath)
            # print(srcapath)
            # print(srcbpath)
