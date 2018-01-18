#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/1/18 10:43 AM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : config.py

import argparse
from easydict import EasyDict as edict

__C = edict()
cfg = __C

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Reproduce: Look, Listen and Learn')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size of training (default:16)')
