#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/1/18 10:39 AM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : train.py

from __future__ import print_function

import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from L_3_Pytorch.config import cfg, parse_arguments
from L_3_Pytorch.nets import L_3


def main(args):
    model = L_3()
    model = model.cuda()
    x_vision = Variable(torch.randn((1, 3, 224, 224)).cuda())
    x_audio = Variable(torch.randn((1, 1, 257, 199)).cuda())

    print(model)
    output = model(x_vision, x_audio)

    print(output)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))