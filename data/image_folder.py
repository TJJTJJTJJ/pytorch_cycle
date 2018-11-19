#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-14 下午4:44
# Author : TJJ
# 作者是直接改的源码,我就需要什么改什么,作者是真牛
import os

__all__ = ['make_dataset']

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename):
    """
    judge filename is or not image file
    :param filename:
    :return: bool
    """
    return any(filename.endswith(exten) for exten in IMG_EXTENSIONS)

def make_dataset(dir):
    """
    根据dir,取出dir及其子文件夹下的图片
    :param dir: dir
    :return: images list path
    """
    images = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
