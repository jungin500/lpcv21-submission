#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exps.base import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- model config ---------------- #
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.125

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.random_size = (13, 23)
        self.data_dir = "/workspace/ballPerson/dataset/"
        self.train_ann = "lpcv_train.json"
        self.val_ann = "lpcv_val.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False # nano

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 5000
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 100
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (384, 640)
        self.test_conf = 0.3
        self.nmsthre = 0.45

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=False)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=False)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
