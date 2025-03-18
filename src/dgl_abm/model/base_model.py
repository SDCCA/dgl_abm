"""
Base model class of dgl_abm including common methods
"""
import copy
import dgl
import logging
import torch

from dgl_abm.config import Config, CONFIG

from dgl_abm.model.abc_model import ABCModel

logger = logging.getLogger(__name__)

# Set the seed of the random number generator
# this is global and will affect all random number generators
generator = torch.manual_seed(0)



class BaseModel(ABCModel):
    """
    Base class to provide graph based agent based modeling functionality
    """

    def __init__(self,*,model_indentifier, restart=False, savestate=1)
        """
        Create a base model instance or restore from a saved state.
        Checks whether a model indentifier has been specified

        param: model_identifier: str, required. Identifier for the model. Used to save and load model states.
        param: restart: boolean, optional. If True, the model is run from last
        saved step. Default False.
        param: savestate: int, optional. If provided, the model state is saved
        on this frequency. Default is 1 i.e. every time step.
        """

        self.model_indentifier = model_indentifier
        self.restart = restart
        self.savestate = savestate

    def create_model_graph(self):
        """
        create intial graph representing the model, i.e. agents as nodes and their connections as edges.
        Makes use of model specifications in configuration.
        """

        model_graph = network_creation(self.number_agents, self.initial_graph_type, seed=1)
        self.model_graph = model_graph


