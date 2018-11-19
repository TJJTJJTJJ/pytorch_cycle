#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-7 下午5:22
# Author : TJJ

from data.base_dataset import BaseDataset
from data.base_data_loader import BaseDataLoader

import importlib
import torch.utils.data

__all__ = ['get_option_setter', 'CreateDataLoader']

"""
help:
data_loader = CreateDataLoader()
dataset = data_loader.load_data()
for i, data in enumerate(dataset):
    pass
"""

def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In {}.py, there should be a subclass of BaseDataset with class name that matches {} in lowercase.".format(dataset_filename, target_dataset_name))
        exit(0)

    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    """
    create dataset instance
    :param opt:
    :return: instance : class's instance
    """
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print('dataset {} was created'.format(instance.name()))
    return instance

class CustomDatasetDataLoader(BaseDataLoader):
    """
    调用方式
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    dataset_size = len(data_loader)
    dataset = data_loader.load_data()
    for i, data in enumerate(dataset):
        pass

    这里实现了在实际循环取数据时,根据load_data返回的数据类来迭代,
    相当于在dataloader上又封装了一次.
    """
    def name(self):
        return('CustomDatasetDataLoader')

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
