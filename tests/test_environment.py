"""Tests for the environment module."""
import pytest
import torch
import numpy as np
from src.dgl_abm.environment.grid_creation import (grid_creation,
                                                grid_creation_3d, 
                                                GridEnvironment)
from src.dgl_abm.environment.grid_assignment import grid_assignment,grid_assignment_3d
from src.dgl_abm.network.network_creation import network_creation
from src.dgl_abm.environment.grid_update import update_grid, update_grid_3d

def test_basic_grid_creation():
    """Test the 'basic' method of grid_creation function."""
    kwargs = {
        'method': 'basic',
        'x': 7,
        'y': 10
    }
    
    result = grid_creation(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (7, 10), "Grid shape should be 7x10"
    assert torch.all(result.grid_tensor == 1.0), "All grid values should be 1.0"


def test_basic_grid_creation_one_square():
    one_grid = grid_creation(method='basic', x=1, y=1)
    assert one_grid.grid_tensor.shape == (1, 1), "Expected shape is (1, 1)"


def test_distribution_grid_creation():
    """Test the 'distribution' method of grid_creation function."""
    kwargs = {
        'method': 'distribution',
        'x': 5,
        'y': 15,
        'grid_properties': {
            'propertyA': {
                'distribution_type': 'uniform',
                'parameters': [0.0, 3.0],
                'rounding': True,
                'decimals': 2
            },
            'propertyB': {
                'distribution_type': 'normal',
                'parameters': [0.0, 1.0],
                'rounding': True,
                'decimals': 2
            }
        }
    }
    
    result = grid_creation(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (5, 15, 2), "Grid shape should be 5x15x2"
    assert "propertyA" in result.property_to_index, "'propertyA'should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1

    propertyA_values = result.grid_tensor[:, :, 0]
    propertyB_values = result.grid_tensor[:, :, 1]
    assert torch.all(propertyA_values >= 0) and torch.all(propertyA_values <= 3)
    assert torch.mean(propertyB_values).item() == pytest.approx(0, abs=0.1)
    assert torch.std(propertyB_values).item() == pytest.approx(1, abs=0.1)


@pytest.fixture
def example_np_file(tmp_path):
    """Create an example .np file for import."""
    example_grid = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    file_path = tmp_path / "example_grid.npy"
    np.save(file_path, example_grid)
    return file_path

@pytest.fixture
def example_pt_file(tmp_path):
    """Create an example .pt file for import."""
    example_grid = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    file_path = tmp_path / "example_grid.pt"
    torch.save(example_grid, file_path)
    return file_path

def test_custom_import_np(example_np_file):
    """Test the 'custom_import' method with a .np file."""
    kwargs = {
        'method': 'custom_import',
        'path': str(example_np_file),
        'grid_properties': {'propertyA': 0, 'propertyB': 1}
    }
    
    result = grid_creation(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (2, 3, 2), "Grid shape should be 2x3x2"
    assert "propertyA" in result.property_to_index, "'propertyA'should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1
    assert torch.equal(result.grid_tensor[:, :, 0], 
                       torch.tensor([[1, 3, 5], [7, 9, 11]]))
    assert torch.equal(result.grid_tensor[:, :, 1], 
                       torch.tensor([[2, 4, 6], [8, 10, 12]]))


def test_custom_import_pt(example_pt_file):
    """Test the 'custom_import' method with a .pt file."""
    kwargs = {
        'method': 'custom_import',
        'path': str(example_pt_file),
        'grid_properties': {'propertyA': 0, 'propertyB': 1}
    }
    
    result = grid_creation(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (2, 3, 2), "Grid shape should be 2x3x2"
    assert "propertyA" in result.property_to_index, "'propertyA' should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1
    assert torch.equal(result.grid_tensor[:, :, 0], 
                       torch.tensor([[1, 3, 5], [7, 9, 11]]))
    assert torch.equal(result.grid_tensor[:, :, 1], 
                       torch.tensor([[2, 4, 6], [8, 10, 12]]))
    
@pytest.fixture
def example_grid_2d(example_pt_file):
    kwargs = {
        'method': 'custom_import',
        'path': str(example_pt_file),
        'grid_properties': {'propertyA': 0, 'propertyB': 1}
    }
    
    result = grid_creation(**kwargs)
    return result

def test_get_slice(example_grid_2d):
    """Test the get_slice method of GridEnvironment."""
    slice_A = example_grid_2d.get_slice('propertyA')
    assert torch.equal(slice_A, torch.tensor([[1, 3, 5], [7, 9, 11]]))






def test_basic_grid_creation_3d():
    """Test the 'basic' method of grid_creation_3d function."""
    kwargs = {
        'method': 'basic',
        'x': 2,
        'y': 3,
        'z': 4
    }
    
    result = grid_creation_3d(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (2,3,4), "Grid shape should be 2x3x4"
    assert torch.all(result.grid_tensor == 1.0), "All grid values should be 1.0"


def test_basic_grid_creation_one_cube():
    one_grid = grid_creation_3d(method='basic', x=1, y=1, z=1)
    assert one_grid.grid_tensor.shape == (1, 1, 1), "Expected shape is (1, 1, 1)"

def test_distribution_grid_creation_3d():
    """Test the 'distribution' method of grid_creation_3d function."""
    kwargs = {
        'method': 'distribution',
        'x': 2,
        'y': 3,
        'z': 4,
        'grid_properties': {
            'propertyA': {
                'distribution_type': 'uniform',
                'parameters': [0.0, 3.0],
                'rounding': True,
                'decimals': 2
            },
            'propertyB': {
                'distribution_type': 'normal',
                'parameters': [0.0, 1.0],
                'rounding': True,
                'decimals': 2
            }
        }
    }
    
    result = grid_creation_3d(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (2, 3, 4, 2), "Grid shape should be 2x3x4x2"
    assert "propertyA" in result.property_to_index, "'propertyA'should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1

    propertyA_values = result.grid_tensor[:, :, :, 0]
    propertyB_values = result.grid_tensor[:, :, :, 1]
    assert torch.all(propertyA_values >= 0) and torch.all(propertyA_values <= 3)
    assert torch.mean(propertyB_values).item() == pytest.approx(0, abs=0.1)
    assert torch.std(propertyB_values).item() == pytest.approx(1, abs=0.1)


@pytest.fixture
def example_np_file_3d(tmp_path):
    """Create an example 3_D .np file for import."""
    example_grid = np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
                              [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
                              [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]])
    file_path = tmp_path / "example_grid.npy"
    np.save(file_path, example_grid)
    return file_path

@pytest.fixture
def example_pt_file_3d(tmp_path):
    """Create an example 3-D .pt file for import."""
    example_grid = torch.tensor([[[[1, 2], [3, 4], [5, 6]], 
                                  [[7, 8], [9, 10], [11, 12]]],
                              [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
                              [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]])
    file_path = tmp_path / "example_grid.pt"
    torch.save(example_grid, file_path)
    return file_path

def test_custom_import_np_3d(example_np_file_3d):
    """Test the 'custom_import' method with a 3-D .np file."""
    kwargs = {
        'method': 'custom_import',
        'path': str(example_np_file_3d),
        'grid_properties': {'propertyA': 0, 'propertyB': 1}
    }
    
    result = grid_creation_3d(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (3, 2, 3, 2), "Grid shape should be 3x2x3x2"
    assert "propertyA" in result.property_to_index, "'propertyA'should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1
    assert torch.equal(result.grid_tensor[:, :, :, 0], 
                       torch.tensor([[[1, 3, 5], [7, 9, 11]],
                                     [[1, 3, 5], [7, 9, 11]],
                                     [[1, 3, 5], [7, 9, 11]]]))
    assert torch.equal(result.grid_tensor[:, :, :, 1], 
                       torch.tensor([[[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]]]))


def test_custom_import_pt_3d(example_pt_file_3d):
    """Test the 'custom_import' method with a 3-D .pt file."""
    kwargs = {
        'method': 'custom_import',
        'path': str(example_pt_file_3d),
        'grid_properties': {'propertyA': 0, 'propertyB': 1}
    }
    
    result = grid_creation_3d(**kwargs)
    
    assert isinstance(result, GridEnvironment), "Should return GridEnvironment object"
    assert result.grid_tensor.shape == (3, 2, 3, 2), "Grid shape should be 3x2x3x2"
    assert "propertyA" in result.property_to_index, "'propertyA'should be found"
    assert "propertyB" in result.property_to_index, "'propertyB' should be found"
    assert result.property_to_index["propertyA"] == 0
    assert result.property_to_index["propertyB"] == 1
    assert torch.equal(result.grid_tensor[:, :, :, 0], 
                       torch.tensor([[[1, 3, 5], [7, 9, 11]],
                                     [[1, 3, 5], [7, 9, 11]],
                                     [[1, 3, 5], [7, 9, 11]]]))
    assert torch.equal(result.grid_tensor[:, :, :, 1], 
                       torch.tensor([[[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]]]))

@pytest.fixture
def example_distribution_grid(tmp_path):
    """Create an example grid for agent placement. 
    
    Note: x is rows and y is columns.
    """
    example_grid = torch.tensor([[[1], [5]], 
                                 [[1], [1]]])
    file_path = tmp_path / "example_dist_grid.pt"
    torch.save(example_grid, file_path)
    grid=grid_creation(method='custom_import', path=str(file_path), 
                       grid_properties={'propertyA': 0})
    return grid

@pytest.fixture
def example_graph():
    """Create an example graph of agents."""
    agent_graph = network_creation(10,'barabasi-albert')
    return agent_graph

@pytest.fixture
def example_np_position_file(tmp_path):
    """Create an example .npy file of agent positions for import."""
    example_positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1],[0, 1],
                                    [0, 1], [0, 1], [0, 1], [0, 1],[0, 1]])
    file_path = tmp_path / "example_positions.npy"
    np.save(file_path,example_positions)
    return file_path

@pytest.fixture
def example_pt_position_file(tmp_path):
    """Create an example .pt file of agent positions for import."""
    example_positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1],[0, 1],
                                      [0, 1], [0, 1], [0, 1], [0, 1],[0, 1]])
    file_path = tmp_path / "example_positions.pt"
    torch.save(example_positions, file_path)
    return file_path

def test_random_grid_assignment(example_graph,example_distribution_grid):
    """Test the 'random' method of grid_assignment function."""

    grid_assignment(example_graph, example_distribution_grid, method='random')

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)

def test_property_grid_assignment(example_graph,example_distribution_grid):
    """Test the 'grid_property' method of grid_assignment function."""

    grid_assignment(example_graph, example_distribution_grid, method='grid_property', 
                    grid_property='propertyA')

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))
    
def test_custom_import_grid_assignment_pt(example_graph,example_distribution_grid,
                                          example_pt_position_file):
    """Test the 'custom_import' method of grid_assignment function."""

    grid_assignment(example_graph, example_distribution_grid, method='custom_import', 
                    path=str(example_pt_position_file))

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))

def test_custom_import_grid_assignment_np(example_graph,example_distribution_grid,
                                          example_np_position_file):
    """Test the 'custom_import' method of grid_assignment function."""

    grid_assignment(example_graph, example_distribution_grid, method='custom_import', 
                    path=str(example_np_position_file))

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))
 
@pytest.fixture
def example_distribution_grid_3d(tmp_path):
    """Create an example grid for 3-D agent placement. 
    
    Note: x is rows and y is columns.
    """
    example_grid = torch.tensor([[[[1], [10]], 
                                 [[1], [1]]],
                                [[[1], [1]], 
                                 [[1], [1]]]])
    file_path = tmp_path / "example_dist_grid_3d.pt"
    torch.save(example_grid, file_path)
    grid=grid_creation(method='custom_import', path=str(file_path), 
                       grid_properties={'propertyA': 0})
    return grid

@pytest.fixture
def example_np_3d_position_file(tmp_path):
    """Create an example .npy file of agent positions for import."""
    example_positions = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1],[0, 1, 1],
                                [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],[0, 1, 1]])
    file_path = tmp_path / "example_positions_3d.npy"
    np.save(file_path,example_positions)
    return file_path

@pytest.fixture
def example_pt_3d_position_file(tmp_path):
    """Create an example .pt file of agent positions for import."""
    example_positions = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 0], 
                                      [1, 1, 1],[0, 1, 1],[0, 1, 1], [0, 1, 1], 
                                      [0, 1, 1], [0, 1, 1],[0, 1, 1]])
    file_path = tmp_path / "example_positions_3d.pt"
    torch.save(example_positions, file_path)
    return file_path


def test_random_grid_assignment_3d(example_graph,example_distribution_grid_3d):
    """Test the 'random' method of grid_assignment function."""

    grid_assignment_3d(example_graph, example_distribution_grid_3d, method='random')

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert torch.all(example_graph.ndata['z'] >= 0)
    assert torch.all(example_graph.ndata['z'] < 2)

def test_property_grid_assignment_3d(example_graph,example_distribution_grid_3d):
    """Test the 'property' method of grid_assignment function."""

    grid_assignment_3d(example_graph, example_distribution_grid_3d, method='grid_property', 
                    grid_property='propertyA')
 
    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert torch.all(example_graph.ndata['z'] >= 0)
    assert torch.all(example_graph.ndata['z'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))
    assert (torch.sum(example_graph.ndata['z'] == 0) > 
            torch.sum(example_graph.ndata['z'] == 1))
    
def test_custom_import_grid_assignment_pt_3d(example_graph,example_distribution_grid_3d,
                                          example_pt_3d_position_file):
    """Test the 'custom_import' method of grid_assignment function."""

    grid_assignment_3d(example_graph, example_distribution_grid_3d, 
                       method='custom_import', path=str(example_pt_3d_position_file))

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert torch.all(example_graph.ndata['z'] >= 0)
    assert torch.all(example_graph.ndata['z'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))
    assert (torch.sum(example_graph.ndata['z'] == 0) < 
            torch.sum(example_graph.ndata['z'] == 1))

def test_custom_import_grid_assignment_np_3d(example_graph,example_distribution_grid_3d,
                                          example_np_3d_position_file):
    """Test the 'custom_import' method of grid_assignment function."""

    grid_assignment_3d(example_graph, example_distribution_grid_3d, 
                       method='custom_import', path=str(example_np_3d_position_file))

    assert torch.all(example_graph.ndata['x'] >= 0) 
    assert torch.all(example_graph.ndata['x'] < 2)
    assert torch.all(example_graph.ndata['y'] >= 0)
    assert torch.all(example_graph.ndata['y'] < 2)
    assert (torch.sum(example_graph.ndata['x'] == 0) > 
            torch.sum(example_graph.ndata['x'] == 1))
    assert (torch.sum(example_graph.ndata['y'] == 0) < 
            torch.sum(example_graph.ndata['y'] == 1))
    assert (torch.sum(example_graph.ndata['z'] == 0) < 
            torch.sum(example_graph.ndata['z'] == 1))

def test_update_grid_noise():
    """Test the 'noise' method of the update_grid function."""
    grid = torch.zeros(2, 5, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "2D")
    
    grid_properties = {
        'propertyA': {
            'count': 3,
            'distribution': {
                'distribution_type': 'random',
                'parameters': [0.0, 3.0],
                'rounding': True,
                'decimals': 2
            }
        },
        'propertyB': {
            'ratio': 0.5,
            'distribution': {
                'distribution_type': 'normal',
                'parameters': [1.0, 0.1],
                'rounding': True,
                'decimals': 2
            }
        }
    }
    
    update_grid(grid_environment, method='noise', grid_properties=grid_properties)
    
    propertyA_values = grid_environment.grid_tensor[:, :, 0]
    propertyB_values = grid_environment.grid_tensor[:, :, 1]
    assert propertyA_values.mean() > 0
    assert torch.all(propertyA_values) < 3
    assert propertyB_values.mean() == pytest.approx(0.5, abs=0.1)
    assert torch.all(propertyB_values <= 2)

def test_update_grid_targeted():
    """Test the 'targeted' method of the update_grid function."""
    grid = torch.zeros(2, 5, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "PropertyB": 1}, "2D")
    
    updates = torch.tensor([[0, 0, 0, 1], [1, 4, 1, 2]])
    
    update_grid(grid_environment, method='targeted', updates=updates)
    
    assert grid_environment.grid_tensor[0, 0, 0] == 1
    assert grid_environment.grid_tensor[1, 4, 1] == 2


def test_update_grid_custom_import_np(example_np_file):
    """Test the 'custom_import' method of the update_grid function."""
    grid = torch.zeros(2, 3, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "2D")

    update_grid(grid_environment, method='custom_import', grid_properties =
                            {'propertyB': {"path": str(example_np_file), 
                                           "reference_layer": 1}})
    assert torch.equal(grid_environment.grid_tensor[:, :, 1], 
                       torch.tensor([[2, 4, 6], [8, 10, 12]]))
 
def test_update_grid_custom_import_pt(example_pt_file):
    """Test the 'custom_import' method of the update_grid function."""
    grid = torch.zeros(2, 3, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "2D")

    update_grid(grid_environment, method='custom_import', grid_properties =
                            {'propertyB': {"path": str(example_pt_file), 
                                           "reference_layer": 1}})
    assert torch.equal(grid_environment.grid_tensor[:, :, 1],
                       torch.tensor([[2, 4, 6], [8, 10, 12]]))
    

def test_update_grid_3d_noise():
    """Test the 'noise' method of the update_grid_3d function."""
    grid = torch.zeros(2, 5, 2, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "3D")
    grid_properties = {
        'propertyA': {
            'count': 3,
            'distribution': {
                'distribution_type': 'random',
                'parameters': [0.0, 3.0],
                'rounding': True,
                'decimals': 2
            }
        },
        'propertyB': {
            'ratio': 0.5,
            'distribution': {
                'distribution_type': 'normal',
                'parameters': [1.0, 0.1],
                'rounding': True,
                'decimals': 2
            }
        }
    }
    
    update_grid_3d(grid_environment, method='noise', grid_properties=grid_properties)
    
    propertyA_values = grid_environment.grid_tensor[:, :, :, 0]
    propertyB_values = grid_environment.grid_tensor[:, :, :, 1]
    assert propertyA_values.mean() > 0
    assert torch.all(propertyA_values) < 3
    assert propertyB_values.mean() == pytest.approx(0.5, abs=0.1)
    assert torch.all(propertyB_values <= 2)

def test_update_grid_3d_targeted():
    """Test the 'targeted' method of the update_grid_3d function."""
    grid = torch.zeros(2, 5, 2, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "PropertyB": 1}, "3D")
    
    updates = torch.tensor([[0, 0, 0, 0, 1], [1, 4, 1, 1, 2]])
    
    update_grid_3d(grid_environment, method='targeted', updates=updates)
    
    assert grid_environment.grid_tensor[0, 0, 0, 0] == 1
    assert grid_environment.grid_tensor[1, 4, 1, 1] == 2


def test_update_grid_3d_custom_import_np(example_np_file_3d):
    """Test the 'custom_import' method of the update_grid_3d function."""
    grid = torch.zeros(3, 2, 3, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "3D")

    update_grid_3d(grid_environment, method='custom_import', grid_properties =
                            {'propertyB': {"path": str(example_np_file_3d), 
                                           "reference_layer": 1}})
    assert torch.equal(grid_environment.grid_tensor[:, :, :, 1],
                       torch.tensor([[[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]]]))
 
def test_update_grid_3d_custom_import_pt(example_pt_file_3d):
    """Test the 'custom_import' method of the update_grid_3d function."""
    grid = torch.zeros(3, 2, 3, 2)
    grid_environment = GridEnvironment(grid, {"propertyA": 0, "propertyB": 1}, "3D")

    update_grid_3d(grid_environment, method='custom_import', grid_properties =
                            {'propertyB': {"path": str(example_pt_file_3d), 
                                           "reference_layer": 1}})
    assert torch.equal(grid_environment.grid_tensor[:, :, :, 1],
                       torch.tensor([[[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]],
                                     [[2, 4, 6], [8, 10, 12]]]))
    