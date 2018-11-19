#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-14 下午3:54
# Author : TJJ

from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset

import random
from PIL import Image
import os

__all_ = ['UnalignedDataset', 'modify_commandline_options', 'initialize', 'name']

class UnalignedDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase+'B')

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """
        __getitem__的返回值可以是list,tuple,dict
        :param index:
        :return:
        """
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size-1)
        B_path = self.B_paths[index_B]

        if self.opt.direction == 'BtoA' :
            input_nc, output_nc = self.opt.output_nc, self.opt.input_nc
        else:
            input_nc, output_nc = self.opt.input_nc, self.opt.output_nc

        if input_nc == 1:
            A_img = Image.open(A_path).convert('L')
        else:
            A_img = Image.open(A_path).convert('RGB')

        if output_nc == 1:
            B_img = Image.open(A_path).convert('L')
        else:
            B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'






