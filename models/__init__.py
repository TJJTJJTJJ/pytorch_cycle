#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-11-13 下午4:37
# Author : TJJ

"""
1. 可以看到,这个文件独立于具体模型
2. 具体模型的命名:model_name_model.py  modelnamemodel
   eg. model_name:cycle_gan---->cycle_gan_model.py---->CycleGANModel
3. 具体的模型必须继承basemdoel
"""

"""
base_options.gather_options----get_option_setter----model.modify_commandline_options
create_model----find_model_using_name
           |----CycleGANModel.initialize----Basemodel.initialize
"""

__all__ = ['create_model', 'get_option_setter']



import importlib
from .base_model import BaseModel

def find_model_using_name(model_name):
    """
    根据model_name导入具体模型'models/model_name_model.py'
    :param model_name: eg. cycle_gan
    :return: mdoel class eg.cycle_gan_model.CycleGANModle
    """
    # step1 import 'models/model_name_model'
    model_filename = 'models.'+model_name+'_model'
    modellib = importlib.import_module(model_filename)

    # step2 get model_name
    model = None
    target_model_name = model_name.replace('_','')+'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print_str = "In {model_filename}.py, there should be a subclass of BaseModel with class name " \
              "that matches {target_model_name} in lowercase.".format(model_filename=model_filename, \
                                                                      target_model_name=target_model_name)
        print(print_str)
        exit(0)

    return model

def get_option_setter(model_name):
    """
    为base_options.gather_option提供model的option参数

    :param model_name:
    :return:
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model {} has been create".format(instance.name()))

    return instance
