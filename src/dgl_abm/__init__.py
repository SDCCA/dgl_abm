"""Documentation about dgl_abm."""

import logging

# from dgl_abm import config
from dgl_abm.model.model import AgentBasedModel

# __all__ = ['config', 'AgentBasedModel']
__all__ = ['AgentBasedModel']

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.1.0"
