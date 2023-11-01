# -*- coding: utf-8 -*-
# (c) 2023 Hanshi Sun LGPL
# utils that used in model operations


import torch
import os
import yaml

from .info import terminal_msg

def get_config(config_path):
    """
    load config from .yaml file
    
    params:
        config_path: config.yaml file path
        
    return:
        config: config dict
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def count_parameters(model):
    """
    count the number of parameters in a network.
    
    params:
        model: model to be manipulated
        
    return:
        params, trainable_params
    """
    num_params = 0
    num_trainable_params = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_trainable_params += p.numel()
    return num_params, num_trainable_params