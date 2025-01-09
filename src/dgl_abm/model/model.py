import copy
import dgl
import torch
import pickle
import logging

from pathlib import Path
from dgl.data.utils import save_graphs, load_graphs

# from dgl_ptm.network.network_creation import network_creation
# from dgl_ptm.model.step import ptm_step
# from dgl_ptm.agentInteraction.weight_update import weight_update
# from dgl_ptm.config import Config, CONFIG
from dgl_abm.util.network_metrics import average_degree

# Set the seed of the random number generator
# this is global and will affect all random number generators
generator = torch.manual_seed(0)

logger = logging.getLogger(__name__)

def sample_distribution_tensor(type, distParameters, nSamples, round=False, decimals=None):
    """
    create and return samples from different distributions

    :param type: Type of distribution to sample
    :param distParameters: array of parameters as required/supported by requested distribution type
    :param nSamples: number of samples to return (as 1d tensor)
    :param round: optional, whether the samples are to be rounded
    :param decimals: optional, required if round is specified. decimal places to round to
    """
    # check if each item in distParameters are torch tensors, if not convert them
    for i, item in enumerate(distParameters):
        # if item has dtype NoneType, raise error
        if item != None and not isinstance(item, torch.Tensor):
                distParameters[i] = torch.tensor(item)

    if not isinstance(nSamples, torch.Tensor):
        nSamples = torch.tensor(nSamples)

    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(distParameters[0], distParameters[1]).sample([nSamples])
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(distParameters[0], distParameters[1]).sample([nSamples])
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(probs=distParameters[0],logits=distParameters[1],validate_args=None).sample([nSamples])
    elif type == 'multinomial':
        dist = torch.gather(torch.Tensor(distParameters[1]), 0, torch.multinomial(torch.tensor(distParameters[0]), nSamples, replacement=True))
    elif type == 'truncnorm':
        # distParameters are mean, standard deviation, min, and max. cdf(x)=(1+erf(x/2^0.5))/2. cdf^-1(x)=2^0.5*erfinv(2*x-1).
        trunc_val_min = (distParameters[2]-distParameters[0])/distParameters[1]
        trunc_val_max = (distParameters-distParameters[0])/distParameters[1]
        cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0))))/2
        cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0))))/2

        uniform_samples = torch.rand(size)
        sample_ppf = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 *(cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1)

        dist = distParameters[0] + distParameters[1] * sample_ppf

    else:
        raise NotImplementedError('Currently only uniform, normal, multinomial, and bernoulli distributions are supported')

    if round:
        if decimals == None:
            raise ValueError('rounding requires decimals of rounding accuracy to be specified')
        else:
            return torch.round(dist,decimals=decimals)
    else:
        return dist

def sample_distribution(distribution, nSamples):
    return sample_distribution_tensor(distribution['type'], distribution['parameters'], nSamples, round = distribution['round'], decimals = distribution['decimals'])


class AgentBasedModel(object):
    """
    Agent based model class
    """

    def __init__(self, *, model_identifier, root_path = '.'):
        """
        Create a new ABM instance.
        Checks whether a model identifier has been specified.

        param: model_identifier: str, required. Identifier for the model. Used to save and load model states.
        param: root_path: str, optional. Root path where to store the model data and states.
        """

        self._model_identifier = model_identifier
        self.root_path = root_path
        self.model_dir = self.root_path / Path(self._model_identifier)
        
        # Step count.
        # Note that the config no longer contains the step count:
        # the config is determined before a starting run;
        # the step count may not be correct when loading a config to continue a run
        # (whether restoring a run after a crash or continuing from a milestone).
        self.step_count = 0

        # Attach config.
        # self.config = copy.deepcopy(CONFIG)
        # self.steering_parameters = self.config.steering_parameters.__dict__
        self.graph = None
        self.step_first = -1

        # Process version.
        version_path = Path(__file__).resolve().parents[2] / 'version.md'
        self.version = version_path.read_text().splitlines()[0]

    def save_model_parameters(self, overwrite = False):
        # Save config to yaml file.
        cfg_filename = f'{self.model_dir}/{self._model_identifier}_{self.step_count}'
        if overwrite: cfg_filename = cfg_filename + '.yaml'
        else: cfg_filename = _make_path_unique(cfg_filename, '.yaml')
        # self.config.to_yaml(cfg_filename)
        # logger.warning(f'The model parameters are saved to {cfg_filename}.')

    def set_model_parameters(self, *, parameterFilePath=None, overwrite = False, **kwargs):
        """
        Load or set model parameters

        :param parameterFlePath: optional, path to parameter file. If not,  default values are used.
        :param **kwargs: flexible passing of mode parameters. Only those supported by the model are accepted.
                         If parameters are passed, non-specifed parameters will be set with defaults.

        """
        # cfg = CONFIG # default values

        # if parameterFilePath:
        #     cfg = Config.from_yaml(parameterFilePath)
        #     if kwargs:
        #         # if both parameterFilePath and kwargs are set, combine them into one.
        #         # if fields are duplicated, kwargs will overwrite parameterFilePath
        #         for key, value in kwargs.items():
        #             if isinstance(value, dict):
        #                 # Special recursive case for steering_parameters:
        #                 # this makes sure to append to, not overwrite, the steering parameters.
        #                 for subkey, subvalue in value.items():
        #                     setattr(cfg.__dict__[key], subkey, subvalue)
        #             else:
        #                 setattr(cfg, key, value)
        #         logger.warning(
        #             'model parameters have been provided via parameterFilePath and **kwargs. '
        #             '**kwargs will overwrite parameterFilePath'
        #             )
        # elif kwargs:
        #     cfg = Config.from_dict(kwargs)

        # if parameterFilePath is None and not kwargs:
        #     logger.warning('no model parameters have been provided, Default values are used')

        # if cfg.model_identifier != self._model_identifier:
        #     logger.warning(f'A model identifier has been set as "{self._model_identifier}". '
        #                    f'But another identifier "{cfg.model_identifier}" is provided (perhaps by default). '
        #                    f'The identifier "{self._model_identifier}" will be used.')

        # cfg.model_identifier = self._model_identifier # see config.py for why cfg.model_identifier

        # # update model parameters/ attributes
        # cfg_dict = cfg.model_dump(by_alias=True, warnings=False)
        # for key, value in cfg_dict.items():
        #     setattr(self.config, key, value)
        # self.steering_parameters = self.config.steering_parameters.__dict__
        
        # Correct the paths
        self.model_dir = self.root_path / Path(self._model_identifier)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # self.steering_parameters['npath'] = str(self.model_dir / Path(self.config.steering_parameters.npath))
        # self.steering_parameters['epath'] = str(self.model_dir / Path(self.config.steering_parameters.epath))

        # Save updated config to yaml file.
        self.save_model_parameters(overwrite)

    def initialize_model(self, restart = False):
        """
        Create network and initialize agent properties in correct order, thereby initializing a model.

        Params:
            restart: boolean or a pair of ints, optional.
            If True, the model is initialized from the last checkpoint,
            if a pair of ints, the model is initialized at that step from that milestone,
                e.g (2,0) would be the first milestone at step 2
                and (2,1) would be the second milestone at step 2.
            Default False.
        """

        self.inputs = None
        if isinstance(restart, bool):
            if restart:
                logger.info(f'Loading model state from checkpoint: {self.model_dir}')
                self.inputs = _load_model(self.model_dir)
        elif isinstance(restart, tuple):
            milestone_dir = None
            if restart[1] == 0:
                milestone_dir = f'{self.model_dir}/milestone_{restart[0]}'
            else:
                milestone_dir = f'{self.model_dir}/milestone_{restart[0]}_{restart[1]}'
            logger.info(f'Loading model state from milestone: {milestone_dir}')
            self.inputs = _load_model(milestone_dir)

        if self.inputs:
            self.graph = copy.deepcopy(self.inputs["graph"])
            self.generator_state = self.inputs["generator_state"]
            generator.set_state(self.generator_state)
            self.step_count = self.inputs["step_count"]
        # else:
        #     torch.manual_seed(self.config.seed)
        
        self.create_network()
        # self.initialize_agent_properties()
        # self.graph = self.graph.to(self.config.device)
        # self.initialize_model_properties()
        # self.steering_parameters['modelTheta'] = self.steering_parameters['modelTheta'].to(self.config.device)

        # store random generator state
        self.generator_state = generator.get_state()

        # number of edges(links) in the network
        # self.number_of_edges = self.graph.number_of_edges()
        self.average_degree = average_degree(self.graph)

    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        # agent_graph = network_creation(self.config.number_agents, self.config.initial_graph_type, **self.config.initial_graph_args.__dict__)
        # #this should fix issues with execution on GPU (fix by VG)
        # self.graph = agent_graph.to(self.config.device)

    def step(self):
        """
        Perform a single step of the model.

        After the step, the current state (graph, generator, step, and version) may be saved:

        The state can be saved with a fixed period (config.checkpoint_period) to keep a restore point in case of a crash.
        Only the newest checkpoint is retained.

        The state can also be saved at specific steps (config.milestones) to store specific (important) states.
        For example, specific states can be stored to start multiple runs from the same state with different parameters going forward.
        All milstones are retained. The first milestone at each time step X is stored in the subdirectory `./milestone_X`;
        any subsequent milestones at the same time step X are stored in the subdirectory `./milestone_X_i` (where i is the instance).
        """
        try:
            print(f'performing step {self.step_count} of {self.config.step_target}')
            # ptm_step(self.graph, self.config.device, self.step_count, self.steering_parameters)

            # number of edges(links) in the network.
            self.number_of_edges = self.graph.number_of_edges()
            self.average_degree = average_degree(self.graph)

        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting.
            # raise RuntimeError(f'execution of step failed for step {self.step_count}')
            print(f'placeholder')

        # save the model state every step reported by checkpoint_period and at specific milestones.
        # checkpoint saves overwrite the previous checkpoint; milestone get unique folders.
        # Note that milestones are not created at the first step of a run;
        # this prevents duplicate saves when running from a milestone.
        first_step = self.step_count == self.step_first
        # save_checkpoint = 0 < self.config.checkpoint_period and self.step_count % self.config.checkpoint_period == 0
        # save_milestone = self.config.milestones and self.step_count in self.config.milestones and not first_step
        # if save_checkpoint or save_milestone:
        #     self.inputs = {
        #         'graph': copy.deepcopy(self.graph),
        #         'generator_state': generator.get_state(),
        #         'step_count': self.step_count,
        #         'process_version': self.version
        #     }

        #     # Note that a sinlge step could be both a checkpoint and a milestone.
        #     # The checkpoint could be necessary to restore a crashed process while
        #     # the milestone is required output.
        #     if save_checkpoint:
        #         _save_model(self.model_dir, self.inputs)
        #     if save_milestone:
        #         milestone_path = _make_path_unique(f'{self.model_dir}/milestone_{self.step_count}')
        #         _save_model(milestone_path, self.inputs)

        self.step_count +=1

    def run(self):
        """run the model for each step until the step_target is reached."""
        # Save config to yaml file.
        self.save_model_parameters()

        self.step_first = self.step_count
        # while self.step_count < self.config.step_target:
        #     self.step()


def _make_path_unique(path, extension = ''):
    """
    Check whether a path already exists and make it unique if it does.

    Paths are made unique by adding "_x" to the path,
    where x is the lowest positive integer for which the path does not exist.

    Params:
        path: the path to make unique
        extension: str, optional, this extension is added to the path
          after any integer added to make the path unique.
          Note that for true extensions, this should start with a dot, e.g ".yaml"
    Returns:
        the modified path, which does not currently exist.
    """
    if Path(f'{path}{extension}').exists():
        instance = 1
        def add_instance(path, instance, extension): return f'{path}_{instance}{extension}'
        while Path(add_instance(path, instance, extension)).exists(): instance += 1
        path = add_instance(path, instance, extension)
    else:
        path = path + extension
    return path

def _save_model(path, inputs):
    """ save the graph, generator_state and process_version in files."""
    Path(path).mkdir(parents=True, exist_ok=True)

    # save the graph with a label
    graph_labels = {'step_count': torch.tensor([inputs["step_count"]])}
    save_graphs(str(Path(path) / "graph.bin"), inputs["graph"], graph_labels)

    # save the generator_state
    with open(Path(path) / "generator_state.bin", 'wb') as file:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], file)

    # save the process version
    with open(Path(path) / "process_version.md", 'w') as file:
        file.writelines([inputs["process_version"] + '\n', f'step={inputs["step_count"]}\n'])


def _load_model(path):
    # Load model graph
    path_graph = Path(path) / "graph.bin"
    if not path_graph.is_file():
        raise ValueError(f'The path {path_graph} is not a file.')

    graph, graph_labels = load_graphs(str(path_graph))
    graph = graph[0]
    graph_step = graph_labels['step_count'].tolist()[0]

    # Load generator_state
    path_generator_state = Path(path) / "generator_state.bin"
    if not path_generator_state.is_file():
        raise ValueError(f'The path {path_generator_state} is not a file.')

    with open(path_generator_state, 'rb') as file:
        generator, generator_step = pickle.load(file)

    # Load process version
    path_process_version = Path(path) / "process_version.md"
    if not path_process_version.is_file():
        raise ValueError(f'The path {path_process_version} is not a file.')

    with open(path_process_version, 'r') as file:
        process_version = file.readlines()[0]

    # Check if graph_step, generator_step and data_step are the same
    if graph_step != generator_step:
        msg = 'The step count in the graph and generator_state are not the same.'
        raise ValueError(msg)
    
    # Check if the saved version and current process version are the same
    version_path = Path(__file__).resolve().parents[2] / 'version.md'
    current_version = version_path.read_text().splitlines()[0]
    if process_version != current_version:
        logger.warning(f'Warning: loading model generated using earlier process version: {process_version}.')

    # Show which step is loaded
    logger.warning(f'Loading model state from step {generator_step}.')

    inputs = {
        'graph': graph,
        'generator_state': generator,
        'step_count': generator_step,
        'process_version': process_version
    }
    return inputs
