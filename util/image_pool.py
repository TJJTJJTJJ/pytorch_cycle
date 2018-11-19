#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-16 上午9:47
# Author : TJJ
import torch
import random

__all__ = ['ImagePool']

class ImagePool():
    """
    为了保证判别器稳定,所以在取图片送入判别器时,以1/2的概率选择当前的图片,1/2的概率选择之前的图片.
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """

        :param images: N*C*H*W
        :return: return_images: N*C*H*W
        pool_size = 0: 返回所有图像
        pool_size > 0 当超过pool_size,1/2的概率从self.images中取,同时更新self.images,1/2的概率直接取images
        ref : https://arxiv.org/pdf/1612.07828.pdf
        """
        if self.pool_size == 0:
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone() # 感觉这里不需要clone也行
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images
