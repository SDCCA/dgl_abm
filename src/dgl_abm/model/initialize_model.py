"""This module contains the model class and functions to initialize the model.

Classes:
- Model: Abstract model class
    - save_model_parameters: Saves model parameters to a .yaml file
    - set_model_parameters: Loads or sets model parameters
    - initialize_model: Initializes a model
    - create_network: Assigns edges between agent nodes
    - initialize_global_properties: Initializes global properties/values of the model
    - initialize_time_step_properties: initializes properties/values for each time step
    - initialize_agent_attributes: Initializes and assigns agent attributes
    - step: Performs a single step of the model
    - run: Runs the model for each step until the step_target is reached

Function(s):
- sample_distribution: Formats distribution arguments
- _make_path_unique: Establishes a unique save path
- _save_model: Saves the model state
- _load_model: Loads a saved model state
"""

import copy
import logging
import pickle
from pathlib import Path
import torch
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs

# from dgl_abm.agentInteraction.weight_update import weight_update
from dgl_abm.model.config import CONFIG
from dgl_abm.model.config import Config

# from dgl_ptm.environment import grid_creation, grid_assignment
# from dgl_abm.model.step import abm_step
from dgl_abm.network.network_creation import network_creation
from dgl_abm.util.network_metrics import average_degree
from dgl_abm.util.network_metrics import average_weighted_degree
from dgl_abm.util.network_metrics import node_degree
from dgl_abm.util.network_metrics import node_weighted_degree
from dgl_abm.util.utils import sample_distribution_tensor

# Set the seed of the random number generator
# this is global and will affect all torch random number generators
generator = torch.manual_seed(0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_distribution(distribution: dict, n_samples: int | list | tuple) -> torch.Tensor:
    """Generate a sample from a distribution.

    Args:
        distribution (dict): dictionary specifying type and parameters of distribution
        n_samples (int|list|tuple): number/shape of samples to draw from the distribution

    Returns:
        torch.Tensor
    """
    return sample_distribution_tensor(
        distribution["type"],
        distribution["parameters"],
        n_samples,
        round=distribution["round"],
        decimals=distribution["decimals"],
    )


class Model:
    """Abstract model class."""

    def __init__(self, experiment_identifier: None | str = None, root_path: str = "."):
        """Initialize the model class.

        Args:
            experiment_identifier(str): an identifier used in paths. Defaults to None.
            root_path(str): path to working directory of the model. Defaults to '.'.
        """
        self.experiment_identifier = experiment_identifier
        self.root_path = root_path
        self.model_dir = self.root_path / Path(self.experiment_identifier)

        # Step count.
        # Note that the config no longer contains the step count:
        # the config is determined before a starting run;
        # the step count may not be correct when loading a config to continue a run
        # (whether restoring a run after a crash or continuing from a milestone).
        self.step_count = 0

        # Attach config.
        self.config = copy.deepcopy(CONFIG)
        self.steering_parameters = self.config.steering_parameters.__dict__
        self.graph = None
        self.step_first = -1

        # Process version.
        version_path = Path(__file__).resolve().parents[2] / "version.md"
        self.version = version_path.read_text().splitlines()[0]

    def save_model_parameters(self, overwrite: bool = False) -> None:
        """Save model parameters to a yaml file.

        Arg:
            overwrite (bool): optional, whether to overwrite existing file. Defaults to false.

        Returns:
            None

        Effects:
            Saves the model parameters to cfg_filename, potentially overwriting
            existing values.
        """
        cfg_filename = f"{self.model_dir}/{self.experiment_identifier}_{self.step_count}"
        cfg_filename = f"{cfg_filename}.yaml" if overwrite else _make_path_unique(cfg_filename, ".yaml")
        self.config.to_yaml(cfg_filename)
        config_file_message = f"The model parameters are saved to {cfg_filename}."
        logger.warning(config_file_message)

    def set_model_parameters(
        self, *, parameter_file_path: None | str = None, overwrite: bool = False, **kwargs: dict
    ) -> None:
        """Load and set model parameters.

        This function starts with the default configuration specified in config.py. Then, if a
        parameter_file_path is valid, it replaces the default configuration. The
        kwargs are then used to selectively replace values for any specified parameters.
        The updated configuration is then saved to a .yaml file.

        Args:
            parameter_file_path (str): optional, path to parameter file. If not,
                default demo values are used.
            overwrite (bool): optional, whether to overwrite existing file. Defaults to false
            **kwargs (dict): flexible passing of mode parameters. Only those supported
                by the model are accepted.
        """
        cfg = CONFIG

        if parameter_file_path:
            cfg = Config.from_yaml(parameter_file_path)
            if kwargs:
                # if both parameter_file_path and kwargs are set, combine them
                # into one. If fields are duplicated, kwargs will overwrite
                # parameter_file_path
                for key, value in kwargs.items():
                    if isinstance(value, dict):
                        # Special recursive case for steering_parameters: this
                        # makes sure to append to, not overwrite, the steering
                        # parameters.
                        for subkey, subvalue in value.items():
                            setattr(cfg.__dict__[key], subkey, subvalue)
                    else:
                        setattr(cfg, key, value)
                logger.warning(
                    "model parameters have been provided via "
                    "parameter_file_path and **kwargs. "
                    "**kwargs will overwrite parameter_file_path",
                )
        elif kwargs:
            cfg = Config.from_dict(kwargs)

        if parameter_file_path is None and not kwargs:
            logger.warning(
                "No model parameters have been provided. Default values are used.",
            )

        if cfg.experiment_identifier != self.experiment_identifier:
            identifier_message = (
                f'An experiment identifier has been set as "{self.experiment_identifier}", '
                f'but the identifier "{cfg.experiment_identifier}" is provided by default. '
                f'The identifier "{self.experiment_identifier}" will be used.'
            )
            logger.warning(identifier_message)

        cfg.experiment_identifier = self.experiment_identifier
        self.config.experiment_identifier = self.experiment_identifier

        # update model parameters/ attributes
        cfg_dict = cfg.model_dump(by_alias=True, warnings=False)
        for key, value in cfg_dict.items():
            setattr(self.config, key, value)
        self.steering_parameters = self.config.steering_parameters.__dict__

        # Correct the paths
        self.model_dir = self.root_path / Path(self.experiment_identifier)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        npath = Path(self.config.steering_parameters.npath)
        self.steering_parameters["npath"] = str(self.model_dir / npath)
        epath = Path(self.config.steering_parameters.epath)
        self.steering_parameters["epath"] = str(self.model_dir / epath)

        # Save updated config to yaml file.
        self.save_model_parameters(overwrite=overwrite)

    def initialize_model(self, restart: bool | tuple = False) -> None:
        """Initialize a model.

        This function assigns agent nodes a network and initializes agent properties
        and global properties in correct order.

        Args:
            restart (bool or tuple[int]): optional, defines checkpoint or milestone
                from which to resume model progress

        Notes: If restart is True, the model is initialized from the last recorded
            checkpoint. If restart is a pair of integers, the model is initialized
            from the milestone specified with (step,instance). E.g, (2,0) represents
            the first instance of a milestone recorded at step 2 and, if multiple
            instances of a milestone exist, (2,1) represents the second instance of a
            milestone recorded at step 2.
        """
        self.inputs = None
        if isinstance(restart, bool):
            if restart:
                restart_message = f"Loading model state from checkpoint: {self.model_dir}"
                logger.info(restart_message)
                self.inputs = _load_model(self.model_dir)
        elif isinstance(restart, tuple):
            milestone_dir = None
            if restart[1] == 0:
                milestone_dir = f"{self.model_dir}/milestone_{restart[0]}"
            else:
                milestone_dir = f"{self.model_dir}/milestone_{restart[0]}_{restart[1]}"
            milestone_message = f"Loading model state from milestone: {milestone_dir}."
            logger.info(milestone_message)
            self.inputs = _load_model(milestone_dir)

        if self.inputs:
            self.graph = copy.deepcopy(self.inputs["graph"])
            self.generator_state = self.inputs["generator_state"]
            generator.set_state(self.generator_state)
            self.step_count = self.inputs["step_count"]
        else:
            torch.manual_seed(self.config.seed)
            seed_message = f"Model torch seed set to {self.config.seed}."
            logger.info(seed_message)

        self.create_network()
        if self.config.spatial:
            self.create_grid()
            self.place_agents()
        self.initialize_global_properties()
        self.initialize_time_step_properties()
        self.initialize_agent_attributes()
        self.graph = self.graph.to(self.config.device)
        network_message = f"{self.graph.number_of_nodes()} agents wereinitialized on {self.graph.device} device."
        logger.info(network_message)
        """
        weight_update(
            self.graph,
            self.config.device,
            self.steering_parameters['homophily_parameter'],
            self.steering_parameters['characteristic_distance'],
            self.steering_parameters['truncation_weight']
            )
        """
        # store random generator state
        self.generator_state = generator.get_state()
        # number of edges(links) in the network
        self.number_of_edges = self.graph.number_of_edges()
        # Network Metrics
        self.average_degree = average_degree(self.graph)
        self.average_weighted_degree = average_weighted_degree(self.graph)
        self.graph.ndata["degree"] = node_degree(self.graph)
        self.graph.ndata["weighted_degree"] = node_weighted_degree(self.graph)

    def create_network(self) -> None:
        """Create intial network connecting agents.

        Makes use of intial graph type specified as model parameter.
        """
        agent_graph = network_creation(
            self.config.number_agents,
            self.config.initial_graph_type,
            **self.config.initial_graph_args.__dict__,
        )
        self.graph = agent_graph

    '''
    def create_grid(self):
        """
        Create an initial grid environment for agents.
        (Optional)
        """
        grid_environment = grid_creation(
            **self.config.spatial_creation_args.__dict__
            )
        self.grid_environment = grid_environment

    def place_agents(self):
        """Place agents on the grid environment."""

        self.graph.ndata['x'] = torch.zeros(self.graph.num_nodes()).float()
        self.graph.ndata['y'] = torch.zeros(self.graph.num_nodes()).float()
        grid_assignment(self.graph, self.grid_environment, **self.config.spatial_assignment_args.__dict__)
    '''

    def initialize_global_properties(self) -> None:
        """Initialize any properties or values necessary to run the model or make calculations.

        Note: Global properties are very flexible in creation and usage; it is more appropriate to assign
        properties associated with each time step using the time_step_properties steering parameter.
        """
        if self.steering_parameters.global_properties is not None:
            for key, value in self.steering_parameters.global_properties.items():
                if isinstance(value, list):
                    self.steering_parameters.global_properties[key] = torch.tensor(value)
                elif isinstance(value, torch.Tensor) and len(value) == self.config.step_target:
                    self.steering_parameters.global_properties[key] = value
                elif isinstance(value, dict):
                    self.steering_parameters.global_properties[key] = sample_distribution(
                        self.steering_parameters.global_properties[key]["distribution"],
                        self.steering_parameters.global_properties[key]["shape"],
                    )
                else:
                    unsupported_message = f"Global property {key} must be a dictionary, list, or torch tensor."
                    raise RuntimeError(unsupported_message)
        else:
            self.steering_parameters["global_properties"] = {}

    def initialize_time_step_properties(self) -> None:
        """Initialize properties for each time step."""
        if self.steering_parameters.time_step_properties is not None:
            for key, value in self.steering_parameters.time_step_properties.items():
                if isinstance(value, list) and len(value) == self.config.step_target:
                    self.steering_parameters[key] = torch.tensor(value).to(self.config.device)
                elif isinstance(value, torch.Tensor) and len(value) == self.config.step_target:
                    self.steering_parameters[key] = value.to(self.config.device)
                elif isinstance(value, dict):
                    if "shape" in value:
                        self.steering_parameters[key] = sample_distribution(
                            self.steering_parameters.time_step_properties[key]["distribution"],
                            [self.config.step_target, *self.steering_parameters.time_step_properties[key]["shape"]],
                        ).to(self.config.device)
                    else:
                        self.steering_parameters[key] = sample_distribution(
                            self.steering_parameters.time_step_properties[key], self.config.step_target
                        ).to(self.config.device)
                else:
                    length_message = (
                        f"Time step property {key} must be a distribution dictionary, "
                        "dictionary of distribution and shape, "
                        f"or a list or torch tensor of length matching the number of "
                        f"steps, {self.config.step_target}."
                    )
                    raise RuntimeError(length_message)
            if (
                self.config.steering_parameters.record_time_step_properties
                and self.config.steering_parameters.record_time_step_properties
                in ["all", "All", True, "TRUE", "true", key]
            ):
                self.config.steering_parameters.record_time_step_properties[f"{key}_value"].append(
                    self.steering_parameters.time_step_properties[key],
                )

    def initialize_agent_attributes(self) -> None:
        """Initialize and assign heterogeneous or individually evolving agent attributes.

        Note: agents are represented as nodes of the model graph.
        Values are initialized as tensors of length corresponding to number of
        agents, with values subsequently being assigned to the nodes.
        """
        for key, value in self.agent_parameters.agent_attributes.items():
            if isinstance(value, list) and len(value) == self.graph.num_nodes():
                self.steering_parameters.agent_attributes[key] = torch.tensor(value).to(self.config.device)
            elif isinstance(value, torch.Tensor) and len(value) == self.graph.num_nodes():
                self.steering_parameters.agent_attributes[key] = value.to(self.config.device)
            elif isinstance(value, dict):
                if "shape" in value:
                    self.steering_parameters[key] = sample_distribution(
                        self.steering_parameters.time_step_properties[key]["distribution"],
                        [self.graph.num_nodes(), *self.steering_parameters.time_step_properties[key]["shape"]],
                    ).to(self.config.device)
                else:
                    self.steering_parameters.agent_attributes[key] = sample_distribution(
                        self.steering_parameters.agent_attributes[key]["distribution"],
                        self.graph.num_nodes(),
                    ).to(self.config.device)
            else:
                length_message = (
                    f"Agent property {key} must be a distribution dictionary, "
                    "dictionary of distribution and shape, or a list "
                    f"or torch tensor of length equal to the number of agents, "
                    f"{self.graph.num_nodes()}."
                )
                raise RuntimeError(length_message)

    '''

    def step(self):
        """Perform a single step of the model.

        Note: After the step, the current state (graph, generator, step, and version)
            may be saved for a checkpoint or milestone as specified in the model
            configuration.

            config.checkpoint_period - The state can be saved with a fixed period to
                keep a restore point in case of a crash. Only the newest checkpoint is
                retained.

            config.milestones - The state can also be saved at specific steps to
                store specific (important) states. For example, specific states can be
                stored to start multiple runs from the same state with different
                parameters going forward. All milstones are retained. The first
                milestone at a given timestep X is stored in the subdirectory
                `./milestone_X`; any subsequent instances of the same milestones at
                timestep X are stored in the subdirectory `./milestone_X_i`
                (where i is the instance).
        """
        try:
            step_message = f'Performing step {self.step_count} of {self.config.step_target}'
            logger.info(step_message)
            #### Generalize space saving
            if self.step_count == 0:
                if (agent_graph.number_of_edges()+self.config['noise_ratio'] *
                    agent_graph.number_of_nodes()+self.config['local_ratio'] *
                    agent_graph.number_of_nodes()<2**32):
                    agent_graph = agent_graph.int()
                    storage_message = f"Agent graph storage type: {agent_graph.idtype}"
                    logger.info(storage_message)


            abm_step(
                self.graph,
                self.config.device,
                self.step_count,
                self.steering_parameters
                )
                # Data can be collected periodically (every X steps) and/or at specified time steps.
            do_periodical_data_collection = (
                self.config.data_collection_period > 0
                and self.step_count % self.config.data_collection_period == 0
                )
            do_specific_data_collection = (
                self.config.data_collection_step_list
                and self.step_count in self.config.data_collection_step_list
                )
            if do_periodical_data_collection or do_specific_data_collection:
                #Data collection and storage
                data_collection(
                    agent_graph,
                    timestep = self.step_count,
                    npath = self.config.npath,
                    epath = self.config.epath,
                    ndata = self.config.ndata,
                    edata = self.config.edata,
                    mode = self.config.mode
                    )

            # number of edges(links) in the network
            self.number_of_edges = self.graph.number_of_edges()
            self.average_degree = average_degree(self.graph)

        except Exception as e:
            # TODO: Add model dump here.
            # Also check against previous save to avoid overwriting
            msg = f'Execution of step failed for step {self.step_count}'
            raise RuntimeError(msg) from e

        # save the model state every step reported by checkpoint_period and at
        # specific milestones.
        # checkpoint saves overwrite the previous checkpoint; milestones get
        # unique folders.
        # Note that milestones are not created at the first step of a run;
        # this prevents duplicate saves when running from a milestone.
        first_step = self.step_count == self.step_first
        save_checkpoint = (
            self.config.checkpoint_period > 0
            and self.step_count % self.config.checkpoint_period == 0
            )
        save_milestone = (
            self.config.milestones
            and self.step_count in self.config.milestones and not first_step
            )
        if save_checkpoint or save_milestone:
            self.inputs = {
                'graph': copy.deepcopy(self.graph),
                'generator_state': generator.get_state(),
                'step_count': self.step_count,
                'process_version': self.version
            }

            # Note that a single step could be both a checkpoint and a milestone.
            # The checkpoint could be necessary to restore a crashed process while
            # the milestone is required output.
            if save_checkpoint:
                _save_model(self.model_dir, self.inputs)
            if save_milestone:
                path = f'{self.model_dir}/milestone_{self.step_count}'
                milestone_path = _make_path_unique(path)
                _save_model(milestone_path, self.inputs)

        self.step_count +=1

    def run(self):
        """Run the model for each step until the step_target is reached."""
        # Save config to yaml file.
        self.save_model_parameters()

        self.step_first = self.step_count
        while self.step_count < self.config.step_target:
            self.step()
    '''


def _make_path_unique(path: str, extension: str = "") -> str:
    """Check whether a path already exists and make it unique if it does.

    Note: Paths are made unique by adding "_x" to the path,
        where x is the lowest positive integer for which the path does not exist.

    Args:
        path (str): the path to make unique
        extension (str): optional, this extension is added to the path
            after any integer added to make the path unique. For true extensions,
            this should start with a dot, e.g ".yaml"
    Returns:
        str: the modified path, which does not currently exist
    """
    if Path(f"{path}{extension}").exists():
        instance = 1

        def add_instance(path: str, instance: int, extension: str) -> str:
            return f"{path}_{instance}{extension}"

        while Path(add_instance(path, instance, extension)).exists():
            instance += 1
        path = add_instance(path, instance, extension)
    else:
        path = path + extension
    return path


def _save_model(path: str, inputs: dict) -> None:
    """Save the graph, generator_state and process_version in files.

    Args:
        path (str): path to save the model state files
        inputs (dict): dictionary with the graph, generator_state, step_count and
            process_version
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    # save the graph with a label
    graph_labels = {"step_count": torch.tensor([inputs["step_count"]])}
    save_graphs(str(Path(path) / "graph.bin"), inputs["graph"], graph_labels)

    # save the generator_state
    with Path.open(Path(path) / "generator_state.bin", "wb") as file:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], file)

    # save the process version
    with Path.open(Path(path) / "process_version.md", "w") as file:
        file.writelines(
            [inputs["process_version"] + "\n", f"step={inputs['step_count']}\n"],
        )


def _load_model(path: str) -> dict:
    """Load the graph, generator_state and process_version from files.

    Arg:
        path (str): path to the model state files
    Returns:
        dict: dictionary with the loaded graph, generator_state, step_count and
            process_version
    """
    path_graph = Path(path) / "graph.bin"
    if not path_graph.is_file():
        path_message = f"The path {path_graph} is not a file."
        raise ValueError(path_message)

    graph, graph_labels = load_graphs(str(path_graph))
    graph = graph[0]
    graph_step = graph_labels["step_count"].tolist()[0]

    # Load generator_state
    path_generator_state = Path(path) / "generator_state.bin"
    if not path_generator_state.is_file():
        path_message = f"The path {path_generator_state} is not a file."
        raise ValueError(path_message)

    with Path.open(path_generator_state, "rb") as file:
        generator, generator_step = pickle.load(file)

    # Load process version
    path_process_version = Path(path) / "process_version.md"
    if not path_process_version.is_file():
        path_message = f"The path {path_process_version} is not a file."
        raise ValueError(path_message)

    with Path.open(path_process_version) as file:
        process_version = file.readlines()[0]

    # Check if graph_step, generator_step and data_step are the same
    if graph_step != generator_step:
        msg = "The step count in the graph and generator_state are not the same."
        raise ValueError(msg)

    # Check if the saved version and current process version are the same
    version_path = Path(__file__).resolve().parents[2] / "version.md"
    current_version = version_path.read_text().splitlines()[0]
    if process_version != current_version:
        version_message = f"Warning: loading model generated using earlier process version: {process_version}."
        logger.warning(version_message)

    # Show which step is loaded
    step_message = f"Loading model state from step {generator_step}."
    logger.warning(step_message)

    return {
        "graph": graph,
        "generator_state": generator,
        "step_count": generator_step,
        "process_version": process_version,
    }
