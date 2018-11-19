#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-13 下午5:27
# Author : TJJ
import torch
import os
from collections import OrderedDict

from . import networks

__all__ = ['BaseModel', 'name', 'initialize', 'set_input', 'eval', 'test', 'optimize_parameters',
           'update_learning_rate', 'get_image_paths', 'get_current_visuals',
           'get_current_losses', 'save_networks', 'load_networks']

"""
eval(): model中子model的命名规则, e.g. model_names = ['G_A'], self.netG_A = ...
get_current_losses(): model中loss的命名规则, e.g. loss_names = ['D_A'], self.loss_D_A = ...
loss_names, model_names, visual_names, image_paths这些都是由子类实现提供的
这里实现了nn.Module模型定义与训练分离,即模型定义用专门的类写,训练过程用其他的类写

调用方法：

instance = model()
instance.initialize(opt) # init　model, criterion, optim, 
instance.setup(opt) # schedulers, load
# train
for epoch in range():
    for i, data in enumerate(dataset):
        instance.set_input(data) # data
        instance.optimize_patameters()
    instance.update_learning_rate()
# test
instance.eval()
for i, data in enumerate(dataset):
    instance.set_input(data) # data
    instance.test()

"""

class BaseModel():

    # modify parser if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        由base_options.gather_options调用
        :param parser:
        :param is_train:
        :return: parser
        """
        return parser

    def name(self):
        return "BaseModel"

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths =[]

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def setup(self, opt, parser=None):
        """
        load and print networks
        create schedulers
        :param opt:
        :param parser:
        :return:
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.print_networks(opt.verbose)

        return None

    def eval(self):
        """
        make models eval mode during test time
        :return: None
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net'+name)
                net.eval()

        return None

    def test(self):
        """
        don't need backprop during test time
        :return:
        """
        with torch.no_grad():
            self.forward()

        return None

    def optimize_parameters(self):
        """
        model中参数的更新
        :return:
        """
        pass

    def update_learning_rate(self):
        """
        model中学习率的更新
        :return:
        """
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate={:.7f}'.format(lr))

    def get_image_paths(self):
        """
        return image_paths
        :return:
        """
        return self.image_paths

    def get_current_visuals(self):
        """
        return visualization images to diplay on visdom and save on html
        :return: visual_ret: dict: name:img
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)

        return visual_ret

    def get_current_losses(self):
        """
        return loss to print on console and save to log
        :return: errors_ret: dict: name: loss
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)
        return errors_ret

    def save_networks(self, epoch):
        """
        save model to disk
        :param epoch: str 'last' or int or other model_suffix
        :return: None
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_net_{}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net'+name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

        return None

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """
        加载预训练模型,
        其中涉及到_metadata和InstanceNorm层的加载
        def setup()
        :param epoch: str'last' or int
        :return:
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '{}_net_{}.pth'.format(epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net'+name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from {}'.format(load_path))
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4 这里不是很懂
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

        return None

    def print_networks(self,verbose):
        """
        print networks
        def setup()
        :param verbose: bool: True: print(net) or False: not print(net)
        :return:
        """
        print('-----------------Networks initialized----------------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net'+name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network {name}] Total number of parameters: {num_params:.3f} M'.format(name=name, num_params=num_params/1e6))

    def set_requires_grad(self, nets, requires_grad=False):
        """
        set nets to requires_grad
        fix some nets in train time
        :param nets:  list [torch.nn] or torch.nn
        :param requires_grad: bool
        :return:
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for params in net.parameters():
                    params.requires_grad = requires_grad







