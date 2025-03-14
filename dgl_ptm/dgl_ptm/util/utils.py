"""This module contains miscelleneous utility functions.

Functions:
- load_consumption_model: Load a model from a particular .pth file and assemble
- scale_input: Scale input data according to the input_scale dictionary provided
"""
import os

import torch

from dgl_ptm.util.nn_arch import nn_arch


def load_consumption_model(nn_path,device):
    """Load a model from a particular .pth file and assemble.

    Note: requires structure contained in nn_arch.py
    """
    #print("entered load_consumption_model")
    nn_path=f'{os.getcwd()}{nn_path}'
    modelinfo = torch.load(nn_path, map_location=torch.device(device))


    #print("loaded info")

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    #print("loaded state dict")


    if "cons_scale" in config["data_loader"]["args"]:
        cons_scale=config['data_loader']['args']['cons_scale']
    else:
        cons_scale=1    
    if "i_a_scale" in config["data_loader"]["args"]:
        i_a_scale=config['data_loader']['args']['i_a_scale']
    else:
        i_a_scale=1
    if "input_scale" in config["data_loader"]["args"]:
        input_scale=config['data_loader']['args']['input_scale']
    else:
        input_scale={}


    return model, cons_scale, i_a_scale, input_scale

def scale_input(input_data, scale_dict,inputID="input",verbose=True): #noqa N803
    """Scale input data according to the input_scale dictionary provided."""
    if  scale_dict['dist'] in ["unif", "uniform"]:
        a,b=scale_dict['params']
        if verbose:
            print(f"Scaling requested for {inputID}, Distribution:",scale_dict['dist'],
                  " Parameters:", scale_dict['params'])
        input_data= (input_data - a)/b
        return input_data
    elif  scale_dict['dist'] in ["norm", "normal"]:
        mu, std = scale_dict['params']
        if verbose:
            print(f"Scaling requested for {inputID}, Distribution",scale_dict['dist'],
                  " Parameters:", scale_dict['params'])
        input_data = (input_data - mu)/std
        return input_data
    else:
        print("ERROR: Input scaling was not in recognized format. Neural network "
              "cannot be used as configured.")

