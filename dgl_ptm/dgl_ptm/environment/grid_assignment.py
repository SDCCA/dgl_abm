"""This module allows placement of agents in a spatial environment.

Functions:
- grid_assignment: Assigns agents to positions in a 2D grid environment
- grid_assignment_3d: Assigns agents to positions in a 3D grid environment
"""
from dgl_ptm.util.utils import sample_distribution_tensor
import torch
import numpy as np
import dgl


def grid_assignment(graph, grid_environment, **kwargs):
    '''Assign positions of agents on the grid.
    
    Args:
        graph(DGLGraph): represents the agent network
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            random: 
                This method distributes agents randomly
            property:
                This method distributes agents based on a 
                specified property of the grid environment
                which will be normalized to unity to form 
                an assignment probability. It requires the following 
                keyword argument,
                property (str): property of the grid environment to be used
                    for agent assignment
            custom_import:
                This method distributes agents based on a 
                array or tensor of positions (D0 agent, D1 [x,y]). 
                It requires the following keyword argument,
                path: path at which .np or .pt file is located
    '''
    if kwargs['method'] == "random":
        graph.ndata['x'] = torch.randint(0, grid_environment.grid_shape[0], 
                                         (graph.num_nodes(),)).float()
        graph.ndata['y'] = torch.randint(0, grid_environment.grid_shape[1], 
                                         (graph.num_nodes(),)).float()

    elif kwargs['method'] == "property":
        property = kwargs['property']
        property_slice = grid_environment[property]
        property_sum = torch.sum(property_slice)
        property_slice = property_slice / property_sum
        property_slice = property_slice.view(-1)
        position = torch.multinomial(property_slice, graph.num_nodes(), 
                                     replacement=True)
        x_len, y_len, property_count = grid_environment.grid_shape
        graph.ndata['x'] = (position // y_len).float()
        graph.ndata['y'] = (position % y_len).float()
    elif kwargs['method'] == "custom_import":
        if 'path' not in kwargs:
            raise ValueError('Path to position tensor must be provided for'
                              '"custom_import" method.')
        path = kwargs['path']
        if path.endswith(('.np', '.npy')):
            position = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            position = torch.load(path)
        else:
            raise ValueError('File type not supported for agent positioning; please '
                                'use ".npy", ".pt", or ".np".')
        if position.size(1) != 2:
            raise ValueError('Position tensor must be 2D with [n_agents, 2] shape.')
        if position.size(0) != graph.num_nodes():
            raise ValueError('Position tensor must have an entry for each of the '
                             f'{graph.num_nodes()} agents. '
                             f'Current shape: {position.size()}')
        graph.ndata['x'] = position[:, 0].float()
        graph.ndata['y'] = position[:, 1].float()

    else:
        raise NotImplementedError('Currently only random, property, and custom_import '
                                  'methods are supported for grid assignment.')
    


def grid_assignment_3d(graph, grid_environment, **kwargs):
    '''Assign positions of agents in the 3D grid.
    
    Args:
        graph(DGLGraph): represents the agent network
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            random: 
                This method distributes agents randomly
            property:
                This method distributes agents based on a 
                specified property of the grid environment
                which will be normalized to unity to form 
                an assignment probability. It requires the following 
                keyword argument,
                property (str): property of the grid environment to be used
                    for agent assignment
            custom_import:
                This method distributes agents based on a 
                array or tensor of positions (D0 agent, D1 [x,y,z]). 
                It requires the following keyword argument,
                path: path at which .np or .pt file is located
    '''
    if kwargs['method'] == "random":
        graph.ndata['x'] = torch.randint(0, grid_environment.grid_shape[0], 
                                         (graph.num_nodes(),)).float()
        graph.ndata['y'] = torch.randint(0, grid_environment.grid_shape[1], 
                                         (graph.num_nodes(),)).float()
        graph.ndata['z'] = torch.randint(0, grid_environment.grid_shape[2], 
                                         (graph.num_nodes(),)).float()

    elif kwargs['method'] == "property":
        property = kwargs['property']
        property_slice = grid_environment[property]
        property_sum = torch.sum(property_slice)
        property_slice = property_slice / property_sum
        property_slice = property_slice.view(-1)
        position = torch.multinomial(property_slice, graph.num_nodes(), 
                                     replacement=True)
        _, y_len, z_len, property_count = grid_environment.grid_shape
        x = (position // (y_len * z_len)).float()
        remainder = position % (y_len * z_len)
        y = (remainder // z_len).float()
        z = (remainder % z_len).float()
        graph.ndata['x'] = x
        graph.ndata['y'] = y
        graph.ndata['z'] = z

    elif kwargs['method'] == "custom_import":
        if 'path' not in kwargs:
            raise ValueError('Path to position tensor must be provided for'
                              '"custom_import" method.')
        path = kwargs['path']
        if path.endswith(('.np', '.npy')):
            position = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            position = torch.load(path)
        else:
            raise ValueError('File type not supported for agent positioning; please '
                                'use ".npy", ".pt", or ".np".')
        if position.size(1) != 3:
            raise ValueError('Position tensor must be of [n_agents, 3] shape.')
        if position.size(0) != graph.num_nodes():
            raise ValueError('Position tensor must have an entry for each of the '
                             f'{graph.num_nodes()} agents. '
                             f'Current shape: {position.size()}')
        graph.ndata['x'] = position[:, 0].float()
        graph.ndata['y'] = position[:, 1].float()
        graph.ndata['z'] = position[:, 2].float()

    else:
        raise NotImplementedError('Currently only random, property, and custom_import '
                                  'methods are supported for grid assignment.')
    
    
