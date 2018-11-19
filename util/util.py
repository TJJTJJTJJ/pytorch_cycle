#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-16 下午9:26
# Author : TJJ
import os
import numpy as np
import ntpath
from PIL import Image
from scipy.misc import imresize

__all__ = ['mkdirs', 'mkdir', 'tensor2im', 'save_image', 'save_images']

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image, imtype=np.uint8):
    """
    取第一个tensor
    :param input_image: N*C*H*W
    :param imtype: np.type
    :return: image_numpy: H*W*C
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# image_numpy to disk
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# web to disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """

    :param webpage:
    :param visuals: dict {name:tensor} name来自model的定义
    :param image_path: str: str来自data的定义,指image的来源路径
    :param aspect_ratio:
    :param width:
    :return:
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '{}_{}.png'.format(name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)