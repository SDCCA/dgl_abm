"""This module pertains to spatial grid creation.

Classes:
- GridEnvironment: Represents a grid environment with properties for each location

Functions:
- grid_creation: Creates a representation of the spatial environment
- grid_creation_3d: Creates a representation of a 3-D spatial environment
"""

import numpy as np
import torch
from dgl_abm.util.utils import sample_distribution_tensor

# grid_creation - Creates a representation of the spatial environment
# within which agents act and interact.


class GridEnvironment:
    """This class represents a spatial grid environment with properties."""

    def __init__(self, grid_tensor: torch.Tensor, property_index: dict, space: str) -> None:
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

    def get_slice(self, grid_property: str) -> torch.Tensor:
        """Return the layer of the grid tensor corresponding to a given property.

        Arg:
            grid_property (str): name of the property for which the slice is requested
        """
        if grid_property not in self.property_to_index:
            property_message = f'Property "{grid_property}" not found.'
            raise KeyError(property_message)
        index = self.property_to_index[grid_property]
        if self.space == "2D":
            dimensions = 2
            if len(self.grid_tensor.shape) == dimensions:
                return self.grid_tensor
            return self.grid_tensor[:, :, index]
        if self.space == "3D":
            dimensions = 3
            if len(self.grid_tensor.shape) == dimensions:
                return self.grid_tensor
            return self.grid_tensor[:, :, :, index]
        space_message = 'Grid environment space must be either "2D" or "3D".'
        raise ValueError(space_message)

    def __getitem__(self, grid_property: str) -> torch.Tensor:
        """Return the layer of the grid tensor corresponding to a given property.

        Arg:
            grid_property (str): name of the property for which the index is requested
        """
        return self.get_slice(grid_property)


def grid_creation(method: str, **kwargs: dict) -> GridEnvironment:
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
                grid_properties (dict): dictionary of property distribution dictionaries
                    to be assigned to the grid (format example,
                    {"property1": {"distribution_type": "uniform", "parameters": [0, 1],
                            "rounding": False, "decimals": None}})
            custom_import:
                This method creates a grid environment by importing a
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel,
                and requires the following keyword arguments:
                path (str): path at which .np or .pt file is located
                grid_properties (dict): dictionary of property names to be assigned
                    to the third dimension (format example {"property1": 0,
                    "property2": 1})

        kwargs (dict): keyword arguments to be supplied to the grid creation method

    Return:
        GridEnvironment: Grid environment created via the specified method
    """
    if method == "basic":
        x = kwargs["x"]
        y = kwargs["y"]
        grid = torch.ones(x, y)
        return GridEnvironment(grid, {"ones": 0}, "2D")

    if method == "distribution":
        x = kwargs["x"]
        y = kwargs["y"]
        grid_properties = kwargs["grid_properties"]
        grid = torch.zeros(x, y, len(grid_properties))
        n = x * y
        for i, prop in enumerate(grid_properties):
            distribution = grid_properties[prop]
            grid[:, :, i] = sample_distribution_tensor(
                distribution["distribution_type"],
                distribution["parameters"],
                n,
                rounding=distribution["rounding"],
                decimals=distribution["decimals"],
            ).reshape(x, y)
        return GridEnvironment(grid, {key: i for i, key in enumerate(grid_properties.keys())}, "2D")

    if method == "custom_import":
        if "path" not in kwargs:
            value_message = 'Path to grid tensor must be provided for"custom_import" method.'
            raise ValueError(value_message)
        path = kwargs["path"]
        grid_properties = kwargs["grid_properties"]
        if path.endswith((".np", ".npy")):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith(".pt"):
            grid = torch.load(path)
        else:
            value_message = 'File type not supported for grid creation; please use ".npy", ".pt", or ".np".'
            raise ValueError(value_message)
        return GridEnvironment(grid, grid_properties, "2D")
    message = "Unsupported grid creation method received."
    raise NotImplementedError(message)


def grid_creation_3d(method: str, **kwargs: dict) -> GridEnvironment:
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
                grid_properties (dict): dictionary of property distribution dictionaries
                    to be assigned to the grid (format example,
                    {"property1": {"distribution_type": "uniform", "parameters": [0, 1],
                            "rounding": False, "decimals": None}})
            custom_import:
                This method creates a grid environment by importing a
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position in the lattice with a
                value coresponding to each property/channel,
                and requires the following keyword arguments:
                path (str): path at which .np or .pt file is located
                grid_properties (dict): dictionary of property names to be assigned
                    to the third dimension (format example {"property1": 0,
                    "property2": 1})

        kwargs (dict): keyword arguments to be supplied to the grid creation method

    Return:
        GridEnvironment: Grid environment created via the specified method
    """
    if method == "basic":
        x = kwargs["x"]
        y = kwargs["y"]
        z = kwargs["z"]
        grid = torch.ones(x, y, z)
        return GridEnvironment(grid, {"ones": 0}, "3D")

    if method == "distribution":
        x = kwargs["x"]
        y = kwargs["y"]
        z = kwargs["z"]
        grid_properties = kwargs["grid_properties"]
        grid = torch.zeros(x, y, z, len(grid_properties))
        n = x * y * z
        for i, prop in enumerate(grid_properties):
            distribution = grid_properties[prop]
            grid[:, :, :, i] = sample_distribution_tensor(
                distribution["distribution_type"],
                distribution["parameters"],
                n,
                rounding=distribution["rounding"],
                decimals=distribution["decimals"],
            ).reshape(x, y, z)
        return GridEnvironment(grid, {key: i for i, key in enumerate(grid_properties.keys())}, "3D")

    if method == "custom_import":
        if "path" not in kwargs:
            value_message = 'Path to grid tensor must be provided for"custom_import" method.'
            raise ValueError(value_message)
        path = kwargs["path"]
        grid_properties = kwargs["grid_properties"]
        if path.endswith((".np", ".npy")):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith(".pt"):
            grid = torch.load(path)
        else:
            value_message = 'File type not supported for grid creation; please use ".npy", ".pt", or ".np".'
            raise ValueError(value_message)
        return GridEnvironment(grid, grid_properties, "3D")
    message = "Unsupported grid creation method received."
    raise NotImplementedError(message)
