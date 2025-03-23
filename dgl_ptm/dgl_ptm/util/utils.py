"""This module contains miscelleneous utility functions.

Functions:
- sample_distribution_tensor: Acquires samples from different distributions
- load_consumption_model: Loads a model from a particular .pth file and assemble
- scale_input: Scales input data according to the input_scale dictionary provided
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

def sample_distribution_tensor(type, dist_parameters, n_samples,#noqa PLR0912
                                round=False, decimals=None):
    """Generate and return samples from different distributions.

    Args:
        type (str): Type of distribution to sample
        dist_parameters (list): array of parameters as required/supported by
            requested distribution type
        n_samples (int): number of samples to return (as 1d tensor)
        round (bool): optional, whether the samples are to be rounded
        decimals (int): optional, required if round is specified. decimal places to
            round to
    Returns:
        torch.Tensor: samples from the specified distribution 
    """
    # check if each item in dist_parameters are torch tensors, if not convert them
    for i, item in enumerate(dist_parameters):
        # if item has dtype NoneType, raise error
        if item is not None and not isinstance(item, torch.Tensor):
                dist_parameters[i] = torch.tensor(item)

    if not isinstance(n_samples, torch.Tensor):
        n_samples = torch.tensor(n_samples)

    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(
            dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(
            dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(
            probs=dist_parameters[0], logits=dist_parameters[1], validate_args=None
            ).sample([n_samples])
    elif type == 'multinomial':
        multinomial_samples = torch.multinomial(
            torch.tensor(dist_parameters[0]), n_samples, replacement=True
            )
        dist = torch.gather(torch.Tensor(dist_parameters[1]), 0, multinomial_samples)
    elif type == 'truncnorm':
        # dist_parameters are mean, standard deviation, min, and max.
        # cdf(x)=(1+erf(x/2^0.5))/2. cdf^-1(x)=2^0.5*erfinv(2*x-1).
        trunc_val_min = (dist_parameters[2]-dist_parameters[0])/dist_parameters[1]
        trunc_val_max = (dist_parameters-dist_parameters[0])/dist_parameters[1]
        cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0))))/2
        cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0))))/2

        uniform_samples = torch.rand(n_samples)
        inverse_transform = torch.erfinv(
            2 *(cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1
            )
        sample_ppf = torch.sqrt(torch.tensor(2.0)) * inverse_transform

        dist = dist_parameters[0] + dist_parameters[1] * sample_ppf
    elif type == 'beta':
        dist = torch.distributions.beta.Beta(dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    elif type == 'random':
        dist = torch.rand(n_samples)
        if len(dist_parameters) == 0:
            pass
        if len(dist_parameters) == 1:
            dist = dist * (dist_parameters[0])
        elif len(dist_parameters) == 2:
            dist = dist * (dist_parameters[1] - dist_parameters[0]) + dist_parameters[0]
        else:
            raise ValueError('random distribution supports 0, 1-max, or 2-min/max '
                                'parameters')
        
    else:
        raise NotImplementedError(
            'Currently only uniform, normal, multinomial, beta, random, tuncated normal'
            'and bernoulli distributions are supported'
            )

    if round:
        if decimals is None:
            raise ValueError(
                'rounding requires decimals of rounding accuracy to be specified'
                )
        else:
            return torch.round(dist,decimals=decimals)
    else:
        return dist

