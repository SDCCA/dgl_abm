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
from pydantic import RootModel
from pydantic import field_validator

logger = logging.getLogger(__name__)


class DistributionDictEntry(BaseModel):
    """Base class for distribution parameter dictionary entry."""

    type: str = "uniform"
    parameters: list[int | float | list[int | float]] = Field(default_factory=lambda: [0.0, 1.0])
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v, values) -> torch.Tensor | list[torch.Tensor]:  # noqa: N805, ANN001
        if values.get("type") == "multinomial":
            for i in v:
                if not isinstance(i, list):
                    type_message = "Multinomial parameters must be a list of lists."
                    raise TypeError(type_message)
            return [torch.tensor(i) for i in v]
        return torch.tensor(v)

    model_config = ConfigDict(validate_default=True)


class AgentAttributeDict(BaseModel):
    """Base class for agent attribute dictionary."""

    root: dict[str, DistributionDictEntry | torch.Tensor | list] = Field(
        default_factory=lambda: {"DefaultAttribute": DistributionDictEntry()}
    )
    model_config = ConfigDict(validate_default=True)


class GlobalPropertiesDict(BaseModel):
    """Base class for global variable dictionary."""

    root: dict[str, DistributionDictEntry | dict | int | float | list | torch.Tensor] = Field(
        default_factory=lambda: {"DefaultGlobalAttribute": 0}
    )

    @field_validator("root")
    def validate_dict_values(
        cls, v
    ) -> dict[  # noqa: N805, ANN001
        str, DistributionDictEntry | dict | int | float | list | torch.Tensor
    ]:
        """Validate that each value is of an accepted type."""
        for key, value in v.items():
            if isinstance(value, DistributionDictEntry):
                continue
            if isinstance(value, dict):
                required_keys = {"distribution", "shape"}
                if set(value.keys()) != required_keys:
                    message = (
                        f"If value for '{key}' is not an int, float, list, or tensor, "
                        f"it must be a dictionary with keys {required_keys}."
                    )
                    raise ValueError(message)
        return v

    model_config = ConfigDict(validate_default=True)


class TimeStepPropertiesDict(BaseModel):
    """Base class for time step attribute dictionary."""

    root: dict[str, DistributionDictEntry | dict | int | float | list | torch.Tensor] = Field(
        default_factory=lambda: {"DefaultTimeStepAttribute": 0}
    )

    @field_validator("root")
    def validate_dict_values(
        cls, v
    ) -> dict[  # noqa: N805, ANN001
        str, DistributionDictEntry | dict | int | float | list | torch.Tensor
    ]:
        """Validate that each value is of an accepted type."""
        for key, value in v.items():
            if isinstance(value, DistributionDictEntry):
                continue
            if isinstance(value, dict):
                required_keys = {"distribution", "shape"}
                if set(value.keys()) != required_keys:
                    message = (
                        f"If value for '{key}' is not an int, float, list, tensor, "
                        f"or distribution dictionary, it must be a dictionary with "
                        f"keys {required_keys}."
                    )
                    raise ValueError(message)
            else:
                message = (
                    f"Value for '{key}' must be an int, float, list, tensor, distribution "
                    f"dictionary, or a dictionary with keys {required_keys}."
                )
                raise TypeError(message)
        return v

    model_config = ConfigDict(validate_default=True)


class HomophilyDictEntry(BaseModel):
    """Base class for homophily dictionary entry."""

    keys: list[str] = ["wealth"]
    homophily_parameter: int | float = 1.0
    characteristic_distance: int | float = 3.33


class HomophilyDict(RootModel[dict[str, HomophilyDictEntry]]):
    """Base class for homophily dictionary."""

    root: dict[str, HomophilyDictEntry] = Field(default_factory=lambda: {"wealth": HomophilyDictEntry()})
    model_config = ConfigDict(validate_default=True)


class SteeringParams(BaseModel):
    """Base class for steering parameters.

    These are the parameters used within each step of the model.
    """

    edata: list[str] | None = Field(default_factory=lambda: ["all"])
    epath: str = "./edge_data"
    format: str = "xarray"
    mode: str = "w"
    ndata: list[str | list[str | list[str]]] | None = Field(default_factory=lambda: ["all_except", ["a_table"]])
    npath: str = "./agent_data.zarr"
    nn_path: str | None = "default"
    global_properties: GlobalPropertiesDict | None = None
    time_step_properties: TimeStepPropertiesDict | None = None
    agent_attributes: AgentAttributeDict | None = None
    del_method: str | None = None
    del_threshold: int | float | None | Literal["balance"] = None
    noise_ratio: float | None = None
    local_ratio: float | None = None
    truncation_weight: float = 1.0e-10
    step_type: str = "default"
    data_collection_period: int = 1
    data_collection_list: list[int] | None = None

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

    @field_validator("x", "y")
    def _check_xy(cls, values) -> int | None:  # noqa: N805, ANN001
        if values.get("method") in ["basic", "distribution"] and (values.get("x") is None or values.get("y") is None):
            xy_message = "x and y must be integers for basic and distribution methods."
            raise ValueError(xy_message)
        if values.get("method") == "distribution" and values.get("properties") is None:
            missing_properties_message = "Define property(ies) for distribution method."
            raise ValueError(missing_properties_message)
        if values.get("method") == "custom_import" and values.get("path") is None:
            missing_path_message = "Define path to .pt or .np file for custom_import method."
            raise ValueError(missing_path_message)

    model_config = ConfigDict(validate_default=True)


class GridAssignmentParams(BaseModel):
    """Base class for agent to grid assignment arguments."""

    method: str = "random"
    property: str | None = None
    path: str | None = None

    @field_validator("property", "path")
    def _check_property_path(cls, values) -> str | None:  # noqa: N805, ANN001
        if values.get("method") == "property" and (values.get("property") is None or values.get("property") == ""):
            path_message = "Define path to .pt or .np file for custom_import method."
            raise ValueError(path_message)

    @field_validator("path")
    def _check_path(cls, v, values) -> str | None:  # noqa: N805, ANN001
        if values.get("method") == "custom_import" and (v is None or v == ""):
            property_message = "Define grid property name for property method."
            raise ValueError(property_message)

    model_config = ConfigDict(validate_default=True)


class Config(BaseModel):
    """Base class for configuration parameters.

    These are the parameters used by the overarching process.
    """

    # because pydantic does not like underscores
    model_identifier: str = Field("test", alias="_model_identifier")
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
    steering_parameters: SteeringParams = SteeringParams()

    # Make sure pydantic validates the default values
    model_config = ConfigDict(
        validate_default=True,
        protected_namespaces=(),  # because _model is a protected namespace
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )

    @classmethod
    def from_yaml(cls, config_file) -> "Config":  # noqa: ANN001
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
