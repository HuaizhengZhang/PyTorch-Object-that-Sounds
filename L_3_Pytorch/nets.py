#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/1/18 4:48 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=0.001)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=0.001)
        self.pool = nn.MaxPool2d(stride=2, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        return x

class L_3(nn.Module):
    def __init__(self):
        super(L_3, self).__init__()
        self.vision_conv1 = BasicConv2d(3, 64, kernel_size=2)
        self.vision_conv2 = BasicConv2d(64, 128, kernel_size=2)
        self.vision_conv3 = BasicConv2d(128, 256, kernel_size=2)
        self.vision_conv4 = BasicConv2d(256, 512, kernel_size=(28, 28))

        self.audio_conv1 = BasicConv2d(1, 64, kernel_size=2)
        self.audio_conv2 = BasicConv2d(64, 128, kernel_size=2)
        self.audio_conv3 = BasicConv2d(128, 256, kernel_size=2)
        self.audio_conv4 = BasicConv2d(256, 512, kernel_size=(32, 24))

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, v_feature, a_feature):
        x_v = self.vision_conv1(v_feature)
        x_v = self.vision_conv2(x_v)
        x_v = self.vision_conv3(x_v)
        x_v = self.vision_conv4(x_v)

        print(x_v.shape)

        x_a = self.audio_conv1(a_feature)
        x_a = self.audio_conv2(x_a)
        x_a = self.audio_conv3(x_a)
        x_a = self.audio_conv4(x_a)

        print(x_a.shape)

        x = torch.cat((x_v, x_a), 1)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x)

        return x

        