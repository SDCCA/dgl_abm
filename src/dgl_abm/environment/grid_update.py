"""This module contains functions for updating properties of the grid environment.

Functions:
- update_grid: Modify the grid environment property information
- update_grid_3d: Modify the environment property information for a 3-D grid
"""

import numpy as np
import torch
from dgl_abm.environment.grid_creation import GridEnvironment
from src.dgl_abm.util.utils import sample_distribution_tensor


def update_grid(grid_environment: GridEnvironment, method: str, **kwargs: dict) -> None:
    """Update the grid environment properties.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            "noise": Updates random positions with distributed values for each
                property specified
            "targeted":
                This method updates the grid environment with targeted values
                for each property
            "custom_import":
                This method updates a grid environment by importing a
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel.
        kwargs (dict): keyword arguments to be supplied to the grid creation method
    """
    x_len, y_len, _ = grid_environment.grid_shape

    if method == "noise":
        _noise_2d(grid_environment, x_len, y_len, kwargs)

    elif method == "targeted":
        _targeted_2d(grid_environment, x_len, y_len, kwargs)

    elif method == "custom_import":
        _custom_import_2d(grid_environment, kwargs)
    else:
        method_message = f'Grid update method "{method}" not supported.'
        raise ValueError(method_message)


def _noise_2d(grid_environment: GridEnvironment, x_len: int, y_len: int, kwargs: dict) -> None:
    """Update random positions with distributed values for each property specified.

    Args:
    grid_environment (GridEnvironment): represents the spatial environment
    x_len(int): length of x dimension
    y_len(int): length of y dimension
    kwargs(dict):
        grid_properties (dict): dictionary of properties to be updated including either,
            ratio (float): the ratio of grid locations to be updated or
            count (int): the number of grid locations to be updated and
            distribution (dict): a dictionary of the distribution from which
            to draw updated values.
            E.g.,
            {'property1': {'ratio': 0.05, 'distribution': {'distribution_type': 'random',
            'parameters': [0, 1], 'rounding': False, 'decimals': None}})
            Note: Defaults to updating all positions if neither count nor
            ratio are provided
    """
    grid_properties = kwargs["grid_properties"]
    for property_name in grid_properties:
        if "count" in grid_properties[property_name]:
            positions = torch.randperm(x_len * y_len)[: grid_properties[property_name]["count"]]
        elif "ratio" in grid_properties[property_name]:
            num_positions = int(grid_properties[property_name]["ratio"] * (x_len * y_len))
            positions = torch.randperm(x_len * y_len)[:num_positions]
        else:
            positions = torch.arange(x_len * y_len)

        x = (positions // y_len).long()
        y = (positions % y_len).long()
        property_index = torch.full(x.shape, grid_environment.property_to_index[property_name])

        value = sample_distribution_tensor(
            grid_properties[property_name]["distribution"]["distribution_type"],
            grid_properties[property_name]["distribution"]["parameters"],
            positions.size(0),
            rounding=grid_properties[property_name]["distribution"]["rounding"],
            decimals=grid_properties[property_name]["distribution"]["decimals"],
        )
        grid_environment.grid_tensor.index_put_(
            (x, y, property_index), value.to(dtype=grid_environment.grid_tensor.dtype)
        )


def _targeted_2d(grid_environment: GridEnvironment, x_len: int, y_len: int, kwargs: dict) -> None:
    """Update the grid environment with targeted values for each property.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        x_len(int): length of x dimension
        y_len(int): length of y dimension
        kwargs(dict):
            updates (torch.tensor): coordinate and value information
                Each row should contain the x, y, property index, and new value
                E.g., ([[0, 0, 1, 0.9], [1, 0, 0, 0]) would point to 'property2' at
                position (0, 0), updating it to 0.9 and 'property1' at (1, 0),
                updating it to 0.
    """
    updates = kwargs["updates"]
    x_valid = ((updates[:, 0] >= 0) & (updates[:, 0] < x_len)).all()
    y_valid = ((updates[:, 1] >= 0) & (updates[:, 1] < y_len)).all()
    if not x_valid:
        x_message = "Coordinates for x in update_grid() are out of range."
        raise ValueError(x_message)
    if not y_valid:
        y_message = "Coordinates for y in update_grid() are out of range."
        raise ValueError(y_message)

    grid_environment.grid_tensor[updates[:, 0].long(), updates[:, 1].long(), updates[:, 2].long()] = updates[:, 3].to(
        dtype=grid_environment.grid_tensor.dtype
    )


def _custom_import_2d(grid_environment: GridEnvironment, kwargs: dict) -> None:
    """Update a grid environment by importing a numpy array or torch tensor from a file (.np or .pt).

    Each element in the array/tensor represents a position with a
    value corresponding to each property/channel
    kwargs (dict):
        grid_properties (dict): dictionary with property name(s) to be updated as keys and
            dictionaries as values with the following entries
            path (str): path at which .np or .pt file is located
            reference layer (int): the index of the imported tensor to be used in the update
            E.g., {'property1': {path: "path/to/file.pt", "reference_layer": 0}}
    """
    required_dimension = 2 + 1
    if "grid_properties" not in kwargs:
        method_message = (
            'Properties to be updated, corresponding path, and layer must be provided for "custom_import" method.'
        )
        raise ValueError(method_message)
    grid_properties = kwargs["grid_properties"]
    for property_name in grid_properties:
        if "path" not in grid_properties[property_name]:
            path_message = 'Path to grid tensor must be provided for "custom_import" method.'
            raise ValueError(path_message)

    path = grid_properties[property_name]["path"]
    if path.endswith((".np", ".npy")):
        grid = torch.from_numpy(np.load(path))
    elif path.endswith(".pt"):
        grid = torch.load(path)
    else:
        file_message = 'File type not supported for grid update; please use ".npy", ".pt", or ".np".'
        raise ValueError(file_message)
    if grid.dim() != required_dimension:
        shape_message = (
            "The reference layer tensor must be of [x, y] shape for each property. I.e, dimension of import must be 3."
        )
        raise ValueError(shape_message)
    if grid_environment.grid_shape[0:1] != grid.shape[0:1]:
        message = "The reference layer tensor shape must match the grid environment shape."
        raise ValueError(message)
    grid_environment.grid_tensor[:, :, grid_environment.property_to_index[property_name]] = grid[
        :, :, grid_properties[property_name]["reference_layer"]
    ]


def update_grid_3d(grid_environment: GridEnvironment, method: str, **kwargs: dict) -> None:
    """Update 3-D grid environment properties.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        method (str): Currently implementable methods include:
            "noise": Updates random positions with distributed values for each
                property specified
            "targeted":
                This method updates the grid environment with targeted values
                for each property
            "custom_import":
                This method updates a grid environment by importing a
                numpy array or torch tensor from a file (.np or .pt),
                where each element represents a position with a
                value coresponding to each property/channel.
        kwargs (dict): keyword arguments to be supplied to the grid creation method
    """
    x_len, y_len, z_len, _ = grid_environment.grid_shape

    if method == "noise":
        _noise_3d(grid_environment, x_len, y_len, z_len, kwargs)
    elif method == "targeted":
        _targeted_3d(grid_environment, x_len, y_len, z_len, kwargs)
    elif method == "custom_import":
        _custom_import_3d(grid_environment, kwargs)
    else:
        message = f'Grid update method "{method}" not supported.'
        raise ValueError(message)


def _noise_3d(grid_environment: GridEnvironment, x_len: int, y_len: int, z_len: int, kwargs: dict) -> None:
    """Update random positions with distributed values for each property specified.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        x_len(int): length of x dimension
        y_len(int): length of y dimension
        z_len(int): length of z dimension
        kwargs(dict):
            grid_properties (dict): dictionary of properties to be updated including either,
                ratio (float): the ratio of grid locations to be updated or
                count (int): the number of grid locations to be updated and
                distribution (dict): a dictionary of the distribution from which
                to draw updated values.
                E.g.,
                {'property1': {'ratio': 0.05, 'distribution': {'distribution_type': 'random',
                'parameters': [0, 1], 'rounding': False, 'decimals': None}})
                Note: Defaults to updating all positions if neither count nor
                ratio are provided
    """
    grid_properties = kwargs["grid_properties"]
    for property_name in grid_properties:
        if "count" in grid_properties[property_name]:
            positions = torch.randperm(x_len * y_len * z_len)[: grid_properties[property_name]["count"]]
        elif "ratio" in grid_properties[property_name]:
            num_positions = int(grid_properties[property_name]["ratio"] * (x_len * y_len * z_len))
            positions = torch.randperm(x_len * y_len * z_len)[:num_positions]
        else:
            positions = torch.arange(x_len * y_len * z_len)

        x = (positions // (y_len * z_len)).long()
        remainder = positions % (y_len * z_len)
        y = (remainder // z_len).long()
        z = (remainder % z_len).long()
        property_index = torch.full(x.shape, grid_environment.property_to_index[property_name])
        value = sample_distribution_tensor(
            grid_properties[property_name]["distribution"]["distribution_type"],
            grid_properties[property_name]["distribution"]["parameters"],
            positions.size(0),
            rounding=grid_properties[property_name]["distribution"]["rounding"],
            decimals=grid_properties[property_name]["distribution"]["decimals"],
        )
        grid_environment.grid_tensor.index_put_(
            (x, y, z, property_index), value.to(dtype=grid_environment.grid_tensor.dtype)
        )


def _targeted_3d(grid_environment: GridEnvironment, x_len: int, y_len: int, z_len: int, kwargs: dict) -> None:
    """Update the grid environment with targeted values for each property.

    Args:
        grid_environment (GridEnvironment): represents the spatial environment
        x_len(int): length of x dimension
        y_len(int): length of y dimension
        z_len(int): length of z dimension
        kwargs(dict):
            updates (torch.tensor): coordinate and value information
                Each row should contain the x, y, z, property index, and new value
                E.g., ([[0, 0, 0, 1, 0.9], [1, 0, 2, 0, 0]) would point to
                'property2' at position (0, 0, 0), updating it to 0.9 and
                'property1' at (1, 0, 2), updating it to 0.
    """
    updates = kwargs["updates"]
    x_valid = ((updates[:, 0] >= 0) & (updates[:, 0] < x_len)).all()
    y_valid = ((updates[:, 1] >= 0) & (updates[:, 1] < y_len)).all()
    z_valid = ((updates[:, 2] >= 0) & (updates[:, 2] < z_len)).all()
    if not x_valid:
        x_message = "Coordinates for x in update_grid() are out of range."
        raise ValueError(x_message)
    if not y_valid:
        y_message = "Coordinates for y in update_grid() are out of range."
        raise ValueError(y_message)
    if not z_valid:
        z_message = "Coordinates for z in update_grid() are out of range."
        raise ValueError(z_message)

    grid_environment.grid_tensor[
        updates[:, 0].long(), updates[:, 1].long(), updates[:, 2].long(), updates[:, 3].long()
    ] = updates[:, 4].to(dtype=grid_environment.grid_tensor.dtype)


def _custom_import_3d(grid_environment: GridEnvironment, kwargs: dict) -> None:
    """Update a grid environment by importing a numpy array or torch tensor from a file (.np or .pt).

    Each element in the array/tensor represents a position with a
    value corresponding to each property/channel,
    and requires the following keyword argument:
    grid_properties (dict): dictionary with property name(s) to be updated as keys and
        dictionaries as values with the following entries
        path (str): path at which .np or .pt file is located
        reference layer (int): the index of the imported tensor to be used in the update
        E.g., {'property1': {path: "path/to/file.pt", "reference_layer": 0}}
    """
    required_dimension = 3 + 1
    if "grid_properties" not in kwargs:
        property_message = (
            'Properties to be updated, corresponding path, and layer must be provided for "custom_import" method.'
        )
        raise ValueError(property_message)
    grid_properties = kwargs["grid_properties"]
    for property_name in grid_properties:
        if "path" not in grid_properties[property_name]:
            path_message = 'Path to grid tensor must be provided for "custom_import" method.'
            raise ValueError(path_message)

    path = grid_properties[property_name]["path"]
    if path.endswith((".np", ".npy")):
        grid = torch.from_numpy(np.load(path))
    elif path.endswith(".pt"):
        grid = torch.load(path)
    else:
        file_message = 'File type not supported for grid update; please use ".npy", ".pt", or ".np".'
        raise ValueError(file_message)
    if grid.dim() != required_dimension:
        dimension_message = (
            "The reference layer tensor must be of [x, y, z] shape "
            "for each property. I.e, dimension of import must be 4."
        )
        raise ValueError(dimension_message)
    if grid_environment.grid_shape[0:2] != grid.shape[0:2]:
        shape_message = "The reference layer tensor shape must match the grid environment shape."
        raise ValueError(shape_message)
    grid_environment.grid_tensor[:, :, :, grid_environment.property_to_index[property_name]] = grid[
        :, :, :, grid_properties[property_name]["reference_layer"]
    ].to(dtype=grid_environment.grid_tensor.dtype)
