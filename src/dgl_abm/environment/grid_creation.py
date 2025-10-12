"""This module pertains to spatial grid creation.

Classes:
- GridEnvironment: Represents a grid environment with properties for each location

Functions:
- grid_creation: Creates a representation of the spatial environment 
- grid_creation_3d: Creates a representation of a 3-D spatial environment
"""
import numpy as np
import torch

from dgl_ptm.util.utils import sample_distribution_tensor

# grid_creation - Creates a representation of the spatial environment 
# within which agents act and interact.

class GridEnvironment:
    """This class represents a spatial grid environment with properties."""
    def __init__(self, grid_tensor, property_index, space):
        """Initialize the grid environment.

        Args:
            grid_tensor (torch.Tensor): tensor storing the grid environment properties
            property_index (dict): dictionary mapping property names to indices
            space (str): dimensionality of the grid environment (2D or 3D)
        """
        self.grid_tensor = grid_tensor
        self.property_to_index = property_index
        self.space = space
        self.grid_shape = grid_tensor.shape

    def get_slice(self, property):
        """Return the layer of the grid tensor corresponding to a given property.

        Arg:
            property (str): name of the property for which the slice is requested
        """
        if property not in self.property_to_index:
            raise KeyError(f"Property '{property}' not found.")
        index = self.property_to_index[property]
        if self.space == "2D":
            if len(self.grid_tensor.shape) == 2:
                return self.grid_tensor
            else:
                return self.grid_tensor[:, :, index]
        elif self.space == "3D":
            if len(self.grid_tensor.shape) == 3:
                return self.grid_tensor
            else:
                return self.grid_tensor[:, :, :, index]

    def __getitem__(self, property):
        """Return the layer of the grid tensor corresponding to a given property.
        
        Arg: 
            property (str): name of the property for which the index is requested
        """
        return self.get_slice(property)

def grid_creation(method, **kwargs):
    """Create a representation of the model environment.

    Args:
        method (str): Currently implementable methods include:
            basic: 
                This method creates a uniform grid environment and
                requires the following keyword arguments,
                x (int): length of the grid in the x-direction/longitude
                y (int): length of the grid in the y-direction/latitude
            distribution:
                This method creates a grid environment with properties and
                requires the following keyword arguments,
                x (int): length of the grid in the x-direction/longitude
                y (int): length of the grid in the y-direction/latitude
                properties (dict): dictionary of property distribution dictionaries 
                    to be assigned to the grid (format example, 
                    {'property1': {'type': 'uniform', 'parameters': [0, 1], 
                            'round': False, 'decimals': None}})
            custom_import:
                This method creates a grid environment by importing a 
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel,
                and requires the following keyword arguments:
                path (str): path at which .np or .pt file is located
                properties (dict): dictionary of property names to be assigned 
                    to the third dimension (format example {'property1': 0, 
                    'property2': 1})

        kwargs (dict): keyword arguments to be supplied to the grid creation method

    Return:
        GridEnvironment: Grid environment created via the specified method
    """
    if method == 'basic':
        x = kwargs['x']
        y = kwargs['y']
        grid = torch.ones(x, y)
        grid_environment=GridEnvironment(grid, {"ones": 0},"2D")
        return grid_environment
    
    elif method == 'distribution':
        x = kwargs['x']
        y = kwargs['y']
        properties = kwargs['properties']
        grid = torch.zeros(x, y, len(properties))
        n = x * y
        for i, prop in enumerate(properties):
            distribution = properties[prop]
            
            grid[:, :, i] = sample_distribution_tensor(distribution['type'],
                            distribution['parameters'], n, 
                            round = distribution['round'],
                            decimals = distribution['decimals']).reshape(x, y)
        grid_environment=GridEnvironment(grid, {key: i for i, key in enumerate(
                                                                    properties.keys())},
                                                                                "2D")
        return grid_environment
           
    elif method == 'custom_import':
        if 'path' not in kwargs:
            raise ValueError('Path to grid tensor must be provided for'
                              '"custom_import" method.')
        path = kwargs['path']
        properties = kwargs['properties']
        if path.endswith(('.np','.npy')):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError('File type not supported for grid creation; please use '
                                '".npy", ".pt", or ".np".')
        grid_environment=GridEnvironment(grid, properties,"2D")
        return grid_environment
            
    else:
        raise NotImplementedError("Unsupported grid creation method received.")



def grid_creation_3d(method, **kwargs):
    """Create a representation of the model environment in three dimensions.

    Args:
        method (str): Currently implementable methods include:
            basic: 
                This method creates a uniform grid environment and
                requires the following keyword arguments,
                x (int): length of the grid in the x-direction/longitude
                y (int): length of the grid in the y-direction/latitude
                z (int): length of the grid in the z-direction/depth
            distribution:
                This method creates a grid environment with properties and
                requires the following keyword arguments,
                x (int): length of the grid in the x-direction/longitude
                y (int): length of the grid in the y-direction/latitude
                z (int): length of the grid in the z-direction/depth
                properties (dict): dictionary of property distribution dictionaries 
                    to be assigned to the grid (format example, 
                    {'property1': {'type': 'uniform', 'parameters': [0, 1], 
                            'round': False, 'decimals': None}})
            custom_import:
                This method creates a grid environment by importing a 
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position in the lattice with a
                value coresponding to each property/channel,
                and requires the following keyword arguments:
                path (str): path at which .np or .pt file is located
                properties (dict): dictionary of property names to be assigned 
                    to the third dimension (format example {'property1': 0, 
                    'property2': 1})

        kwargs (dict): keyword arguments to be supplied to the grid creation method

    Return:
        GridEnvironment: Grid environment created via the specified method
    """
    if method == 'basic':
        x = kwargs['x']
        y = kwargs['y']
        z = kwargs['z']
        grid = torch.ones(x, y, z)
        grid_environment=GridEnvironment(grid, {"ones": 0},"3D")
        return grid_environment
    
    elif method == 'distribution':
        x = kwargs['x']
        y = kwargs['y']
        z = kwargs['z']
        properties = kwargs['properties']
        grid = torch.zeros(x, y, z, len(properties))
        n = x * y * z
        for i, prop in enumerate(properties):
            distribution = properties[prop]
            
            grid[:, :, :, i] = sample_distribution_tensor(distribution['type'],
                            distribution['parameters'], n, 
                            round = distribution['round'],
                            decimals = distribution['decimals']).reshape(x, y, z)
        grid_environment=GridEnvironment(grid, {key: i for i, key in enumerate(
                                                                    properties.keys())},
                                                                                "3D")
        return grid_environment
           
    elif method == 'custom_import':
        if 'path' not in kwargs:
            raise ValueError('Path to grid tensor must be provided for'
                              '"custom_import" method.')
        path = kwargs['path']
        properties = kwargs['properties']
        if path.endswith(('.np','.npy')):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError('File type not supported for grid creation; please use '
                                '".npy", ".pt", or ".np".')
        grid_environment=GridEnvironment(grid, properties,"3D")
        return grid_environment
            
    else:
        raise NotImplementedError("Unsupported grid creation method received.")



