#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-16 下午9:15
# Author : TJJ

from .util import mkdirs, tensor2im, save_image
from .html import HTML

import os
import time
import numpy as np
import sys

__all__ = ['Visualier', 'reset', 'display_current_results', 'plot_current_losses', 'print_current_losses']

"""
self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
self.img_dir = os.path.join(self.web_dir, 'images')

img_path = os.path.join(self.img_dir, 'epoch{:.3f}_{}.png'.format(epoch, label))
save_image(image_numpy, img_path)
"""

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualier():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False

        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port,
                                     env=opt.display_env, raise_exceptions=True, use_incoming_socket=False)
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory {}'.format(self.web_dir))
            mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, 'a') as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss {} ================\n'.format(now))

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) '
              'for displaying training progress.\nYou can suppress connection to Visdom using the '
              'option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start '
              'the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    def display_current_results(self, visuals, epoch, save_result):
        """

        :param visuals: dict {name:imgs, name:imgs}: imgs: N*C*H*W
        :param epoch: int
        :param save_result: bool
        :return:
        """

        # visdom
        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                          table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                          table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                          </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    label_html_row += '<td>{}</td>'.format(label)
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>{}</tr>'.format(label_html_row)
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose(2, 0, 1)) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>{}</tr>'.format(label_html_row)

                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + 'images'))

                    label_html = '<table>{}</table>'.format(label_html)
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + 'labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        # html

        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch{:.3f}_{}.png'.format(epoch, label))
                save_image(image_numpy, img_path)

            webpage = HTML(self.web_dir, 'Experiment name={}'.format(self.name), reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch {}'.format(n))
                ims, txts, links = [], [], []

                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    img_path = 'epoch_{:.3f}_{}.png'.format(n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)

            webpage.save()


    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        """
        self.plot_data['Y']的组织形式是对每一个epoch+couter_ratio,把所有losses的值记录一遍
        e.g. 共有两个loss,分别是loss1, loss2,
        self.plot_data['Y']= [[0.3, 3],
                              [0.2, 2],
                              [0.1, 1],
        self.plot_data['X'] = [1, 1.1, 1.3]
        其中0.2和0.1是loss1的值,2和1是loss2的值

        self.vis.line: Y: N*M, M条线,每条线有N个值
        X : N 或者 N*M


        :param epoch: int
        :param counter_ratio: 0-1
        :param opt:
        :param losses: dict {name:loss, name:loss}
        :return:
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[], 'Y':[], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch+counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        x = self.plot_data['X']
        y = self.plot_data['Y']
        num_loss = len(self.plot_data['legend'])
        try:
            self.vis.line(
                X=np.stack([np.array(x)] * num_loss, axis=1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name+'loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                },
                win=self.display_id
            )
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()


    def print_current_losses(self, epoch,  i, losses, t, t_data):
        """

        :param epoch: ind
        :param i: 当前epoch中的第几个数据 batch_size*iter
        :param losses: dict{name: loss}
        :param t: 当前epoch的当前iter中的batc_size个数据, 每个数据的平均时间,
        :param t_data: 当前epoch,从开始到现在的时间
        :return:
        """
        message = '(epoch: {}, iter: {}, time: {:.3f}, data: {:.3f})'.format(epoch, i, t, t_data)
        for k, v in losses.items():
            message += '{}: {:.3f}'.format(k, v)
        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('{}\n'.format(message))









