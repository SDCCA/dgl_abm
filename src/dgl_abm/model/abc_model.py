"""
Abstract base class for dgl_abm model
"""

class ABCModel(object):
    """
    Abstract model class for graph based agent based modelling
    """

    def __init__(self):
        self._model_identifier=None

    def create_network(self):
        raise NotImplementedError('network creation is not implemented for this class.')
    
    def step(self):
        raise NotImplementedError('step function is not implemented for this class.')
    
    def run(self):
        raise NotImplementedError('run method is not implemented for this class.')
    
