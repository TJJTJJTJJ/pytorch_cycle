#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-16 下午8:36
# Author : TJJ
# 这个模型应该是在测试的时候用,即只有G
#
from .base_model import BaseModel
from .networks import define_G

__all__ = ['TestModel', 'name', 'modify_commandline_options', 'initialize', 'set_input']

class TestModel(BaseModel):

    def name(self):
        return "TestModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.set_defaults(dataset_mode = 'single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir,[epoch]_net_G[model_suffix].pth '
                                 'will be loaded as the generator of TestModel')
        return parser

    def initialize(self, opt):
        assert not is_train, 'TestModel cannot be used in train mode'
        BaseModel.initialize(opt)
        self.loss_names = []
        self.visual_names = ['real_A', 'fake_B']
        self.model_names = ['G'+opt.model_suffix]

        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        setattr(self, 'netG'+opt.model_suffix, self.netG)

    def set_input(self, input):
        self.real_A = inpt['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
