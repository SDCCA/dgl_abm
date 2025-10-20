"""This module contains the step function for the demo model.

Function(s):
- abm_step: Time-stepping module for the demo model
"""

import torch
from dgl import DGLGraph


def abm_step(agent_graph: DGLGraph, device: torch.device, timestep: int, parameters: dict) -> None:
    """Execute step for the demo model.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device (torch.device): Device on which to perform computations
        timestep (int): Current time step
        parameters (dict): Steering parameters specified in model configuration and initialization

    Effects: Completes updates to agent_graph after one step of functional manipulation
    """
    if parameters["step_type"] in ["default", "null", "empty"]:
        agent_graph.edata["weight"] = torch.full((agent_graph.num_edges(),), timestep, device=device)
