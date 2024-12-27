# -*- coding:utf-8 -*-
# @Author: Luo Ge
# @Email: geyouguang@163.com
# @File : CDMIF.py
# @Time : 2024/11/18

import torch
import torch.nn as nn
import torch.nn.functional as F


# *-----------------*
# hyper-parameters
decom_num_layers = 5
kernel_size = 5
filter_num = 50
# *-----------------*

class Decompose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decompose, self).__init__()
        self.num_layers = decom_num_layers
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.layer_ini = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                   kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_ini.weight.data)
        self.theta_ini = nn.Parameter(torch.Tensor([0.01]))

        # 使用 ModuleList 来注册子模块
        self.layer_down = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.theta_list = nn.ParameterList()

        for i in range(self.num_layers):
            down_conv = nn.Conv2d(in_channels=self.out_channel, out_channels=self.in_channel,
                                  kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1, bias=False)
            nn.init.xavier_uniform_(down_conv.weight.data)
            self.layer_down.append(down_conv)

            up_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1, bias=False)
            nn.init.xavier_uniform_(up_conv.weight.data)
            self.layer_up.append(up_conv)

            theta = nn.Parameter(torch.Tensor([0.01]))
            self.theta_list.append(theta)

    def forward(self, X):
        # print('X.shape', X.shape) # torch.Size([1, 64, 64, 6])
        X = X.permute(0, 3, 1, 2)  # transform (B x H x W x 2C) to (B x 2C x H x W)
        p0 = self.layer_ini(X)  # W_e * x
        tensor = torch.mul(torch.sign(p0), F.relu(
            torch.abs(p0) - self.theta_ini))  # T = h_theta(W_e * x), Now T is shape of (C x 3B x H x W)

        for i in range(self.num_layers):
            p1 = self.layer_down[i](tensor)  # W_d * T
            p2 = p1 - X  # W_d * T - X
            p3 = self.layer_up[i](p2)  # W_e * (W_d * T - X)
            p4 = tensor - p3  # T - W_e * (W_d * T - X)
            tensor = torch.mul(torch.sign(p4),
                               F.relu(torch.abs(p4) - self.theta_list[i]))  # T = h_theta(T - W_e * (W_d * T - X))
        # print('tensor.shape', tensor.shape) # torch.Size([1, 64, 64, 64]) is shape of (B x N x H x W), N is the number of filters
        return tensor


class Fuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fuse, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                              kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, T):
        # T is shape of (B x N x H x W)
        Fused = self.conv(T)  # Now 'Fused' is shape of (B x C x H x W)
        Fused = Fused.permute(0, 2, 3, 1)
        # print('Fused.shape', Fused.shape)  # torch.Size([1, 64, 64, 3]) is shape of (B x H x W x C)
        return Fused


class CDMIF(nn.Module):
    def __init__(self):
        super(CDMIF, self).__init__()
        self.in_channel = 3
        self.out_channel = filter_num
        self.decompose = Decompose(2 * self.in_channel, self.out_channel)
        self.fuse = Fuse(self.out_channel, self.in_channel)

    def forward(self, X1, X2):
        X = torch.cat((X1, X2), dim=3)
        T = self.decompose(X)
        # print('T.shape', T.shape) # torch.Size([1, 64, 64, 64]) is shape of (B x N x H x W)
        Fused = self.fuse(T)  # Now 'Fused' is shape of (C x B x H x W)
        return Fused


if __name__ == '__main__':
    instance = CDMIF()
    # print(instance)
    a = torch.rand([1, 64, 64, 3])
    b = torch.rand([1, 64, 64, 3])
    f = instance(a, b)
    print(a.shape)  # torch.Size([1, 64, 64, 3])
    print(b.shape)  # torch.Size([1, 64, 64, 3])
    print(f.shape)  # torch.Size([1, 64, 64, 3])
