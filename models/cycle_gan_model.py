#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-14 下午8:40
# Author : TJJ

from .base_model import BaseModel
from .networks import define_G, define_D, GANLoss
from util.image_pool import ImagePool

import torch
import itertools

__all__ = ['CycleGANModel', 'initialize', 'set_input', 'optimize_parameters']

class CycleGANModel(BaseModel):

    def name(self):
        return "CycleGANModel"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A-->B-->A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B-->A-->B)')
            parser.add_argument('--lambda_identity', type=float, default=0.1,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of'
                                     ' scaling the weight of the identity mapping loss. For example, if the weight of '
                                     'the identity loss should be 10 times smaller than the weight of the '
                                     'reconstruction loss, please set lambda_identity = 0.1, lambda_idt = lambda_identity*lambda_A')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_B', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']
        # netG_A: AtoB ; netG_B: BtoA
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # D_A: B and G_A(A), D_B: A and G_B(B)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                   use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                   use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # optim
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr = opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr = opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        # data
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
        else:
            self.real_A = input['B'].to(self.device)
            self.real_B = input['A'].to(self.device)

    def forward(self):
        # real_A-->fake_B-->rec_A
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        # real_B-->fake_A-->rec_B
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # loss_list = []

        # Identity loss
        if lambda_idt > 0:
            # G_A(B)==B
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B)*lambda_B*lambda_idt
            # G_B(A)==A
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A)*lambda_A*lambda_idt
        else:
            self.loss_idt_A = 0.0
            self.loss_idt_B = 0.0
        # loss_list+=[self.loss_idt_A, self.loss_idt_B]

        # GAN loss netD_A: G_A(A), netD_B: G_B(B)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # loss_list+=[self.loss_G_A, self.loss_G_B]
        # cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # loss_list+=self.loss_cycle_A, self.loss_cycle_B]
        # combined loss
        # self.loss_G_tmp = sum(loss_list)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        """

        :param netD:
        :param real:
        :param fake:
        :return:
        """
        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # combined
        loss_D = (loss_D_real+loss_D_fake)*0.5
        # backward
        loss_D.backward()

        return loss_D

    def backward_D_A(self):
        # D_A: B and G_A(A), D_B: A and G_B(B)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def optimize_parameters(self):
        """
        forward+backward
        :return:
        """
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()









