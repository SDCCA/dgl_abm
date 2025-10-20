"""Documentation about dgl_abm."""

import logging
from .model.initialize_model import Model

__all__ = ["Model"]

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Victoria M. Garibay"
__email__ = "v.m.garibay@uva.nl"
__version__ = "0.1.0"
