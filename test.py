#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-17 下午9:40
# Author : TJJ

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.html import HTML
from util.util import save_images

import os
"""
test的结果保存在results+name中
"""
def main():
    # step1: opt
    opt = TestOptions().parse()
    # step2: data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    # step3: model
    model = create_model(opt)
    model.setup(opt)
    # step4: web ,在test中不使用visdom,而是使用html
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    title = 'Experiment = {}, Phase = {}, Epoch = {}'.format(opt.name, opt.phase, opt.epoch)
    webpage = HTML(web_dir, title=title)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    for i, data in enumerate(dataset):
        if i > opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_images_paths()
        if i % 5 == 0:
            print('processing {:0>4d}-th image...{}'.format(i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()












if __name__ == '__main__':
    main()