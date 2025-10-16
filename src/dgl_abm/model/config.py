"""Configuration parameters for DGL_PTM model.

The configuration parameters are stored in a pydantic object. The model is
initialized with default values. The default values can be overwritten by
providing a yaml file or a dictionary. The keys and values are validated by
pydantic which is a data validation library.
"""

import logging
from pathlib import Path
from typing import Literal
import torch
import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator

logger = logging.getLogger(__name__)


class DistributionDictEntry(BaseModel):
    """Base class for distribution parameter dictionary entry."""

    type: str = "uniform"
    parameters: list[int | float | list[int | float]] = Field(default_factory=lambda: [0.0, 1.0])
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def convert_parameters(cls, v: object, info: ValidationInfo) -> torch.Tensor | list[torch.Tensor]:
        """Convert parameters to torch.Tensor."""
        if info.data is None:
            info_message = "Cannot validate 'parameters' without 'type' field context."
            raise ValueError(info_message)
        if info.data.get("type") == "multinomial":
            for i in v:
                if not isinstance(i, list):
                    type_message = "Multinomial parameters must be a list of lists."
                    raise TypeError(type_message)
            return [torch.tensor(i, dtype=torch.float32) for i in v]
        return torch.tensor(v, dtype=torch.float32)

    model_config = ConfigDict(validate_default=True, extra="forbid")


class AgentAttributeDict(BaseModel):
    """Base class for agent attribute dictionary."""

    root: dict[str, DistributionDictEntry | torch.Tensor | list] = Field(
        default_factory=lambda: {"DefaultAttribute": DistributionDictEntry()},
    )
    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)

    @field_validator("root", mode="before")
    def check_types(cls, v: object) -> dict[str, DistributionDictEntry | torch.Tensor | list]:
        """Validate that the value for each attribute is of an accepted type."""
        if not isinstance(v, dict):
            dictionary_message = (
                "The agent_attributes parameter must be a dictionary mapping "
                "attribute names to a distribution dictionary, torch.Tensor, or list."
            )
            raise TypeError(dictionary_message)
        for key, value in v.items():
            if isinstance(value, (DistributionDictEntry, torch.Tensor, list)):
                continue
            if isinstance(value, dict):
                required_keys = {"distribution", "shape"}
                if set(value.keys()) != required_keys:
                    message = (
                        f"If value for '{key}' is not an int, float, list, or tensor, "
                        f"it must be a dictionary with keys {required_keys}."
                    )
                    raise ValueError(message)
                if not isinstance(value["shape"], (list, tuple)):
                    message = f"The shape for '{key}' must be a list or tuple."
                    raise TypeError(message)
            type_message = f"Value for '{key}' must be DistributionDictEntry, torch.Tensor, or list."
            raise TypeError(type_message)
        return v


class GlobalPropertiesDict(BaseModel):
    """Base class for global variable dictionary."""

    root: dict[str, DistributionDictEntry | dict | int | float | list | torch.Tensor] = Field(
        default_factory=lambda: {"DefaultGlobalAttribute": 0},
    )

    @field_validator("root")
    def validate_dict_values(
        cls,
        v: object,
    ) -> dict[
        str,
        DistributionDictEntry | dict | int | float | list | torch.Tensor,
    ]:
        """Validate that each value is of an accepted type."""
        for key, value in v.items():
            if type(value) in [DistributionDictEntry, int, float, list, torch.Tensor]:
                continue
            if isinstance(value, dict):
                required_keys = {"distribution", "shape"}
                if set(value.keys()) != required_keys:
                    message = (
                        f"If value for '{key}' is not an int, float, list, or tensor, "
                        f"it must be a dictionary with keys {required_keys}."
                    )
                    raise ValueError(message)
                if not isinstance(value["shape"], (list, tuple)):
                    type_message = f"The shape for '{key}' must be a list or tuple."
                    raise TypeError(type_message)
        return v

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)


class TimeStepPropertiesDict(BaseModel):
    """Base class for time step attribute dictionary."""

    root: dict[str, DistributionDictEntry | dict | int | float | list | torch.Tensor] = Field(
        default_factory=lambda: {"DefaultTimeStepAttribute": 0},
    )

    @field_validator("root", mode="before")
    def validate_dict_values(
        cls,
        v: object,
    ) -> dict[
        str,
        DistributionDictEntry | dict | int | float | list | torch.Tensor,
    ]:
        """Validate that each value is of an accepted type."""
        for key, value in v.items():
            if type(value) in [DistributionDictEntry, int, float, list, torch.Tensor]:
                continue
            required_keys = {"distribution", "shape"}
            if isinstance(value, dict):
                if set(value.keys()) != required_keys:
                    message = (
                        f"If value for '{key}' is not an int, float, list, tensor, "
                        f"or distribution dictionary, it must be a dictionary with "
                        f"keys {required_keys}."
                    )
                    raise ValueError(message)
                if not isinstance(value["shape"], (list, tuple)):
                    message = f"The shape for '{key}' must be a list or tuple."
                    raise TypeError(message)
            else:
                message = (
                    f"Value for '{key}' must be an int, float, list, tensor, distribution "
                    f"dictionary, or a dictionary with keys {required_keys}."
                )
                raise TypeError(message)
        return v

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)


class SteeringParams(BaseModel):
    """Base class for steering parameters.

    These are the parameters used within each step of the model.
    """

    nn_path: str | None = None
    global_properties: GlobalPropertiesDict | None = None
    time_step_properties: TimeStepPropertiesDict | None = None
    agent_attributes: AgentAttributeDict | None = None
    deletion_method: str | None = None
    deletion_threshold: int | float | None | Literal["balance"] = None
    noise_ratio: float | None = None
    local_ratio: float | None = None
    truncation_weight: float = 1.0e-10
    step_type: str = "default"

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default=True)


class InitialGraphArgs(BaseModel):
    """Base class for initial graph arguments."""

    seed: int = 1
    new_node_edges: int = 1

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default=True)


class GridCreationParams(BaseModel):
    """Base class for grid creation arguments."""

    method: str = "basic"
    x: int | None = 10
    y: int | None = 10
    properties: dict | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _check_xy(cls, v: object) -> int | None:
        if v.method in ["basic", "distribution"] and (v.x is None or v.y is None):
            xy_message = "x and y must be integers for basic and distribution methods."
            raise ValueError(xy_message)
        if v.method == "distribution" and v.properties is None:
            missing_properties_message = "Define property(ies) for distribution method."
            raise ValueError(missing_properties_message)
        if v.method == "custom_import" and v.path is None:
            missing_path_message = "Define path to .pt or .np file for custom_import method."
            raise ValueError(missing_path_message)
        return v

    model_config = ConfigDict(validate_default=True)


class GridAssignmentParams(BaseModel):
    """Base class for agent to grid assignment arguments."""

    method: str = "random"
    property: str | None = None
    path: str | None = None

    @field_validator("property", "path")
    def check_property(cls, v: object, info: ValidationInfo) -> str | None:
        """Ensure property is defined for property-based method."""
        if info.data is None:
            info_message = "Property presence cannot be validated without method field context."
            raise ValueError(info_message)
        if info.data.get("method") == "property" and (
            info.data.get("property") is None or info.data.get("property") == ""
        ):
            property_message = "Define grid property name for property method."
            raise ValueError(property_message)
        return v

    @field_validator("path")
    def check_path(cls, v: object, info: ValidationInfo) -> str | None:
        """Ensure path is defined for custom import method."""
        if info.data is None:
            info_message = "Custom import path presence cannot be validated without method field context."
            raise ValueError(info_message)
        if info.data.get("method") == "custom_import" and (v is None or v == ""):
            path_message = "Define path to .pt or .np file for custom_import method."
            raise ValueError(path_message)
        return v

    model_config = ConfigDict(validate_default=True)


class Config(BaseModel):
    """Base class for configuration parameters.

    These are the parameters used by the overarching process.
    """

    experiment_identifier: str = "test"
    # Never used to influence processing. This value is meant purely to add a
    # description to identify a parameter setting.
    description: str = ""
    device: str = "cpu"
    seed: int = 42
    number_agents: PositiveInt = 100
    spatial: bool = False
    spatial_creation_args: GridCreationParams = GridCreationParams()
    spatial_assignment_args: GridAssignmentParams = GridAssignmentParams()
    initial_graph_type: str = "barabasi-albert"
    initial_graph_args: InitialGraphArgs = InitialGraphArgs()
    step_target: PositiveInt = 5
    checkpoint_period: int = 10
    milestones: list[PositiveInt] | None = None
    data_collection_period: int = 1
    data_collection_step_list: list[int] | None = None
    edata: list[str] | None = Field(default_factory=lambda: ["all"])
    epath: str = "./edge_data"
    format: str = "xarray"
    mode: str = "w"
    ndata: list[str | list[str | list[str]]] | None = Field(default_factory=lambda: ["all_except", ["a_table"]])
    npath: str = "./agent_data.zarr"
    steering_parameters: SteeringParams = SteeringParams()

    model_config = ConfigDict(
        validate_default=True,
        protected_namespaces=(),
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )

    @classmethod
    def from_yaml(cls, config_file: object) -> "Config":
        """Read configs from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            file_message = f"Config file {config_file} not found."
            raise FileNotFoundError(file_message)

        with Path(config_file).open() as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                parse_message = f"Error parsing config file {config_file}."
                raise SyntaxError(parse_message) from exc
        return cls(**cfg)

    @classmethod
    def from_dict(cls, config_dict) -> "Config":  # noqa: ANN001
        """Read configs from a dict."""
        if not isinstance(config_dict, dict):
            input_message = "Input must be a dictionary."
            raise TypeError(input_message)
        return cls(**config_dict)

    def to_yaml(self, config_file) -> None:  # noqa: ANN001
        """Write configs to a yaml config_file."""
        if Path(config_file).exists():
            overwrite_message = f"Overwriting config file {config_file}."
            logger.warning(overwrite_message)

        cfg = self.model_dump(by_alias=True, warnings=False)

        # if there are tensors, convert them to lists before saving
        def _convert_value(nested_dict) -> dict:  # noqa: ANN001
            """Convert tensors to lists in a nested dictionary."""
            for key, value in nested_dict.items():
                if isinstance(value, torch.Tensor):
                    nested_dict[key] = value.tolist()
                elif isinstance(value, list):
                    nested_dict[key] = [i.tolist() if isinstance(i, torch.Tensor) else i for i in value]
                elif isinstance(value, dict):
                    nested_dict[key] = _convert_value(value)
            return nested_dict

        cfg = _convert_value(cfg)
        with Path(config_file).open("w") as f:
            yaml.dump(cfg, f, sort_keys=False)


CONFIG = Config()
