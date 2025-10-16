"""A test file for config.py."""
import pytest
import yaml
import torch

from src.dgl_abm.model.config import Config
from src.dgl_abm.model.config import DistributionDictEntry, TimeStepPropertiesDict, GlobalPropertiesDict, AgentAttributeDict


@pytest.fixture
def config_parameters():
    return {
        "experiment_identifier": "test_config",
        "number_agents": 100,
        "step_target": 5,
        "steering_parameters": {
            "deletion_method": "probability",
            "deletion_threshold": 0.05,
            "edata": ['all']
            }
    }
@pytest.fixture
def config_file(tmp_path, config_parameters):
    """Return a config file."""
    # change a value
    config_parameters["number_agents"] = 150
    filename = tmp_path / "config.yaml"
    with open(filename , "w") as f:
        yaml.dump(config_parameters, f, sort_keys=False)
    return filename

def test_from_yaml(config_file):
    """Test Config.from_yaml."""
    cfg = Config.from_yaml(config_file)
    assert cfg.experiment_identifier == "test_config"
    assert cfg.number_agents == 150
    assert cfg.step_target == 5

def test_from_dict(config_parameters):
    """Test Config.from_dict."""
    cfg = Config.from_dict(config_parameters)
    assert cfg.experiment_identifier == "test_config"
    assert cfg.number_agents == 100
    assert cfg.step_target == 5

def test_to_yaml(tmp_path):
    """Test Config.to_yaml."""
    cfg = Config()
    cfg.to_yaml(tmp_path / "config.yaml")

    with open(tmp_path / "config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    assert cfg_dict["experiment_identifier"] == "test"
    assert cfg_dict["number_agents"] == 100


def test_defaults():
    """Test that default values are set correctly."""
    cfg = Config()
    assert cfg.experiment_identifier == "test"
    assert cfg.description == ""
    assert cfg.device == "cpu"
    assert cfg.seed == 42
    assert cfg.number_agents == 100
    assert cfg.spatial is False
    assert cfg.initial_graph_type == "barabasi-albert"
    assert cfg.step_target == 5
    assert cfg.checkpoint_period == 10
    assert cfg.milestones is None
    assert cfg.data_collection_period == 1
    assert cfg.data_collection_step_list is None
    assert cfg.edata == ["all"]
    assert cfg.epath == "./edge_data"
    assert cfg.format == "xarray"
    assert cfg.mode == "w"
    assert cfg.ndata == ["all_except", ["a_table"]]
    assert cfg.npath == "./agent_data.zarr"
    assert cfg.steering_parameters.deletion_method == None
    assert cfg.steering_parameters.deletion_threshold == None
    assert cfg.steering_parameters.noise_ratio == None
    assert cfg.steering_parameters.local_ratio == None
    assert cfg.steering_parameters.truncation_weight == 1.0e-10
    assert cfg.steering_parameters.step_type == "default"

def test_invalid_fields(config_parameters):
    """Test that invalid fields are not accepted."""
    config_parameters["invalid_field"] = 100
    with pytest.raises(ValueError):
        _ = Config.from_dict(config_parameters)

def test_agent_attributes():
    valid_dict = {
        "tensor": torch.tensor([1, 2, 3]),
        "list": [1, 2, 3],
        "distribution": DistributionDictEntry(type="uniform", parameters=[0, 1])
    }
    attributes = AgentAttributeDict(root=valid_dict)
    assert isinstance(attributes, AgentAttributeDict)
    assert torch.equal(attributes.root["tensor"], torch.tensor([1, 2, 3]))
    assert attributes.root["list"] == [1, 2, 3]
    assert isinstance(attributes.root["distribution"], DistributionDictEntry)


def test_time_step_properties():
    """Test acceptance of valid values."""
    dictionary = TimeStepPropertiesDict(root={"dist": DistributionDictEntry()})
    assert isinstance(dictionary, TimeStepPropertiesDict)

    dictionary = TimeStepPropertiesDict(root={"custom": {"distribution":{"distribution": {"type": "random", "parameters":[]}, "shape": [10]}, "shape": [2, 3]}})
    assert isinstance(dictionary, TimeStepPropertiesDict)

    dictionary = TimeStepPropertiesDict(root={"int": 1, "float": 2.0, "list": [1, 2], "tensor": torch.tensor([1, 2])})
    assert isinstance(dictionary, TimeStepPropertiesDict)

def test_global_properties():
    """Test acceptance of valid values."""

    example = {
        "integer": 42,
        "float": 1.23,
        "list": [1, 2, 3],
        "tensor": torch.tensor([1, 2, 3]),
        "distribution": DistributionDictEntry(type="uniform", parameters=[0, 1]),
        "dictionary": {"distribution": {"type": "random", "parameters":[]}, "shape": [10]}
    }
    props = GlobalPropertiesDict(root=example)
    assert isinstance(props, GlobalPropertiesDict)
    assert props.root["integer"] == 42
    assert props.root["float"] == 1.23
    assert props.root["list"] == [1, 2, 3]
    assert torch.equal(props.root["tensor"], torch.tensor([1, 2, 3]))
    assert isinstance(props.root["distribution"], DistributionDictEntry)
    assert props.root["dictionary"]["distribution"]["type"] == "random"
    assert props.root["dictionary"]["shape"] == [10]

def test_invalid_values(config_parameters):
    """Test invalid values."""
    config_parameters["number_agents"] = -100
    with pytest.raises(ValueError):
        _ = Config.from_dict(config_parameters)

def test_time_step_invalid_dict_keys():
     """Test dict with no shape key."""
     with pytest.raises(ValueError, match=" must be a dictionary with keys"):
        TimeStepPropertiesDict(root={"invalid_dictionary": {"distribution": {"type": "uniform", "parameters": [0,1]}}})

def test_time_step_invalid_shape_type():
    """Test dict with invalid shape."""
    with pytest.raises(TypeError, match="must be a list or tuple"):
        TimeStepPropertiesDict(root={"invalid_shape": {"distribution": {"type": "uniform", "parameters": [0,1]}, "shape": 4}})

def test_time_step_invalid_value_type():
    """Test dict with invalid value type."""
    with pytest.raises(TypeError, match="must be an int, float, list, tensor"):
        TimeStepPropertiesDict(root={"invalid_type": "all ones"})

def test_global_properties_invalid_dict_keys():
    """Test dict with no shape key."""
    with pytest.raises(ValueError, match="must be a dictionary with keys"):
        GlobalPropertiesDict(root={"invalid_dict": {"distribution": "uniform"}})

def test_global_properties_invalid_shape_type():
    """Test dict with invalid shape."""
    with pytest.raises(TypeError, match="must be a list or tuple"):
        GlobalPropertiesDict(root={"invalid_shape": {"distribution": "uniform", "shape": 4}})

def test_agent_attributes_invalid_value_type():
    """Test dict with invalid value type."""
    with pytest.raises(TypeError, match="must be DistributionDictEntry, torch.Tensor, or"):
        AgentAttributeDict(root={"invalid_type": "all ones"})

def test_distribution_multinomial_conversion():
    distribution = DistributionDictEntry(type="multinomial", parameters=[[0.2, 0.8], [0.5, 0.5]])
    assert isinstance(distribution.parameters, list)
    assert all(isinstance(t, torch.Tensor) for t in distribution.parameters)
    assert distribution.parameters[0].tolist() == pytest.approx([0.2, 0.8], abs=1e-6)

def test_distribution_multinomial_invalid():
    with pytest.raises(TypeError):
        DistributionDictEntry(type="multinomial", parameters=[0.1, 0.9])  
