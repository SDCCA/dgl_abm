"""This module contains functions for updating properties of the grid environment.

Functions:
- update_grid: Modify the grid environment property information
- update_grid_3d: Modify the environment property information for a 3-D grid
"""

import numpy as np
import torch

from dgl_ptm.util.utils import sample_distribution_tensor


def update_grid(grid_environment, method, **kwargs): #noqa PLR0912
    """Update the grid environment properties.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            noise: 
                This method updates random positions in the grid environment 
                with distributed values for each property specified, and requires 
                the following keyword arguments,
                properties (dict): dictionary of properties to be updated including 
                    either, 
                    ratio (float): the ratio of grid locations to be updated or 
                    count (int): the number of grid locations to be updated and
                    distribution (dict): a dictionary of the distribution from which 
                        to draw updated values.
                    E.g., 
                    {'property1': {'ratio': 0.05, 'distribution': {'type': 'random', 
                    'parameters': [0, 1], 'round': False, 'decimals': None}})
                    Note: Defaults to updating all positions if neither count nor 
                    ratio are provided
            targeted:
                This method updates the grid environment with targeted values
                for each property, and requires the following keyword argument:
                updates (torch.tensor): coordinate and value information
                    Each row should contain the x, y, property index, and new value
                    E.g., ([[0, 0, 1, 0.9], [1, 0, 0, 0]) would point to 'property2' at 
                    position (0, 0), updating it to 0.9 and 'property1' at (1, 0), 
                    updating it to 0.
            custom_import:
                This method updates a grid environment by importing a 
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel,
                and requires the following keyword argument:
                properties (dict): dictionary of property name(s) to be updated with a 
                    corresponding dictionary of 
                    path (str): path at which .np or .pt file is located

                    and reference layer the index of the imported tensor to be 
                    used in the update
                    E.g., {'property1': {path: "path/to/file.pt", 'reference_layer': 0}}
        kwargs (dict): keyword arguments to be supplied to the grid creation method"
        """""
    x_len, y_len, property_count = grid_environment.grid_shape

    if method == 'noise':
        properties = kwargs['properties']
        for property in properties.keys():
            if 'count' in properties[property]:
                positions = torch.randperm(x_len * y_len
                                                    )[:properties[property]['count']]
            elif 'ratio' in properties[property]:
                num_positions = int(properties[property]['ratio'] * 
                                    (x_len * y_len))
                positions = torch.randperm(x_len * y_len
                                                    )[:num_positions]
            else:
                positions = torch.arange(x_len * y_len)
            
            x = (positions // y_len).long()
            y = (positions % y_len).long()
            property_index = torch.full(x.shape, 
                                     grid_environment.property_to_index[property])
            
            value = sample_distribution_tensor(
                        properties[property]['distribution']['type'],
                        properties[property]['distribution']['parameters'],
                        positions.size(0), 
                        round=properties[property]['distribution']['round'],
                        decimals=properties[property]['distribution']['decimals'])
            grid_environment.grid_tensor.index_put_((x, y, property_index), 
                                    value.to(dtype=grid_environment.grid_tensor.dtype))
            
    elif method == 'targeted':
        updates = kwargs['updates'] 
        x_valid = ((updates[:, 0] >= 0) & (updates[:, 0] < x_len)).all()
        y_valid = ((updates[:, 1] >= 0) & (updates[:, 1] < y_len)).all()
        if not x_valid:
            raise ValueError("coordinates for x in update_grid() are out of range.")
        if not y_valid:
            raise ValueError("coordinates for y in update_grid() are out of range.")

        grid_environment.grid_tensor[updates[:, 0].long(), updates[:, 1].long(),
                        updates[:, 2].long()] = updates[:, 3].to(
                                            dtype=grid_environment.grid_tensor.dtype)
        
    elif method == 'custom_import':
        if "properties" not in kwargs:
            raise ValueError('Properties to be updated, corresponding path, and layer '
                'must be provided for "custom_import" method.')
        properties = kwargs['properties']
        for property in properties:
            if 'path' not in properties[property]:
                raise ValueError('Path to grid tensor must be provided for'
                              '"custom_import" method.')
        
        path = properties[property]['path']
        if path.endswith(('.np', '.npy')):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError('File type not supported for grid update; please '
                            'use ".npy", ".pt", or ".np".')
        if grid.dim() != 3:
            raise ValueError('The reference layer tensor must be of [x, y] shape '
                                'for each property.')
        if grid_environment.grid_shape[0:1] != grid.shape[0:1]:
            raise ValueError('The reference layer tensor shape must match the grid '
                                'environment shape.')
        grid_environment.grid_tensor[:, :, 
                        grid_environment.property_to_index[property]] = \
                                                    grid[:, :, properties[property]
                                                         ["reference_layer"]]
    else:
        raise ValueError(f'Grid update method "{method}" not supported.') 

def update_grid_3d(grid_environment, method, **kwargs):#noqa PLR0912
    """Update 3-D grid environment properties.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            noise: 
                This method updates random positions in the grid environment 
                with distributed values for each property specified, and requires 
                the following keyword arguments,
                properties (dict): dictionary of properties to be updated including 
                    either, 
                    ratio (float): the ratio of grid locations to be updated or 
                    count (int): the number of grid locations to be updated and
                    distribution (dict): a dictionary of the distribution from which 
                        to draw updated values.
                    E.g., 
                    {'property1': {'ratio': 0.05, 'distribution': {'type': 'random', 
                    'parameters': [0, 1], 'round': False, 'decimals': None}})
                    Note: Defaults to updating all positions if neither count nor 
                    ratio are provided
            targeted:
                This method updates the grid environment with targeted values
                for each property, and requires the following keyword argument:
                updates (torch.tensor): coordinate and value information
                    Each row should contain the x, y, z, property index, and new value
                    E.g., ([[0, 0, 0, 1, 0.9], [1, 0, 2, 0, 0]) would point to 
                    'property2' at position (0, 0, 0), updating it to 0.9 and 
                    'property1' at (1, 0, 2), updating it to 0.
            custom_import:
                This method updates a grid environment by importing a 
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel,
                and requires the following keyword argument:
                properties (dict): dictionary of property name(s) to be updated with a 
                    corresponding dictionary of 
                    path (str): path at which .np or .pt file is located
                    and reference_layer (int): the index of the three dimensional data 
                    in the imported tensor to be used in the update
                    E.g., {'property1': {path: "path/to/file.np", 'reference_layer': 0}}
        kwargs (dict): keyword arguments to be supplied to the grid creation method"
        """""
    x_len, y_len, z_len, property_count = grid_environment.grid_shape

    if method == 'noise':
        properties = kwargs['properties']
        for property in properties.keys():
            if 'count' in properties[property]:
                positions = torch.randperm(x_len * y_len *z_len
                                                    )[:properties[property]['count']]
            elif 'ratio' in properties[property]:
                num_positions = int(properties[property]['ratio'] * 
                                    (x_len * y_len * z_len))
                positions = torch.randperm(x_len * y_len * z_len
                                                    )[:num_positions]
            else:
                positions = torch.arange(x_len * y_len * z_len)

            x = (positions // (y_len * z_len)).long()
            remainder = positions % (y_len * z_len)
            y = (remainder // z_len).long()
            z = (remainder % z_len).long()
            property_index = torch.full(x.shape, 
                                     grid_environment.property_to_index[property])
            value = sample_distribution_tensor(
                        properties[property]['distribution']['type'],
                        properties[property]['distribution']['parameters'],
                        positions.size(0), 
                        round=properties[property]['distribution']['round'],
                        decimals=properties[property]['distribution']['decimals'])
            grid_environment.grid_tensor.index_put_((x, y, z, property_index), 
                                    value.to(dtype=grid_environment.grid_tensor.dtype))
            

    elif method == 'targeted':
        updates = kwargs['updates'] 
        x_valid = ((updates[:, 0] >= 0) & (updates[:, 0] < x_len)).all()
        y_valid = ((updates[:, 1] >= 0) & (updates[:, 1] < y_len)).all()
        z_valid = ((updates[:, 2] >= 0) & (updates[:, 2] < z_len)).all()
        if not x_valid:
            raise ValueError("Coordinates for x in update_grid() are out of range.")
        if not y_valid:
            raise ValueError("Coordinates for y in update_grid() are out of range.")
        if not z_valid:
            raise ValueError("Coordinates for z in update_grid() are out of range.")

        grid_environment.grid_tensor[updates[:, 0].long(), updates[:, 1].long(),
                        updates[:, 2].long(),updates[:, 3].long()] = updates[:, 4].to(
                                            dtype=grid_environment.grid_tensor.dtype)
                
    elif method == 'custom_import':
        if "properties" not in kwargs:
            raise ValueError('Properties to be updated, corresponding path, and layer '
                'must be provided for "custom_import" method.')
        properties = kwargs['properties']
        print(properties)

        for property in properties:
            if 'path' not in properties[property]:
                raise ValueError('Path to grid tensor must be provided for'
                              '"custom_import" method.')
        
        path = properties[property]['path']
        if path.endswith(('.np', '.npy')):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError('File type not supported for grid update; please '
                            'use ".npy", ".pt", or ".np".')
        if grid.dim() != 4:
            raise ValueError('The reference layer tensor must be of [x, y, z] shape '
                                'for each property.')
        if grid_environment.grid_shape[0:2] != grid.shape[0:2]:
            raise ValueError('The reference layer tensor shape must match the grid '
                                'environment shape.')
        grid_environment.grid_tensor[:, :, :,
                        grid_environment.property_to_index[property]] = \
                            grid[:, :, :, properties[property]["reference_layer"]].to(
                                            dtype=grid_environment.grid_tensor.dtype)
    else:
        raise ValueError(f'Grid update method "{method}" not supported.')