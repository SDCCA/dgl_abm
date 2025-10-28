"""This module contains the step function for the demo model.

Function(s):
- abm_step: Time-stepping module for the demo model
"""

import torch
from dgl import DGLGraph
from dgl_abm.network.global_attachment import global_attachment
from dgl_abm.network.link_deletion import link_deletion
from dgl_abm.network.local_attachment import local_attachment
from dgl_abm.network.random_edge_noise import random_edge_noise


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
        # updates agent weight by current timestep, takes no other action unless default
        # steering parameters are altered to include edge manipulations
        agent_graph.edata["weight"] = torch.full((agent_graph.num_edges(),), timestep, device=device)
        initial_edges = agent_graph.num_edges()
        if parameters.get("attachment_ratio") is not None:
            agent_graph = global_attachment(agent_graph, parameters["attachment_ratio"])
        if parameters.get("noise_ratio") is not None:
            random_edge_noise(agent_graph, int(parameters["noise_ratio"] * agent_graph.num_nodes()))
        if parameters.get("local_ratio") is not None:
            agent_graph = local_attachment(
                agent_graph, n_links=int(parameters["local_ratio"] * agent_graph.num_nodes())
            )
        if parameters.get("deletion_method") is not None:
            if parameters["deletion_method"] == "balance":
                deletion_method = "size"
                deletion_threshold = agent_graph.num_edges() - initial_edges
            else:
                deletion_method = parameters["deletion_method"]
                deletion_threshold = parameters["deletion_threshold"]
            link_deletion(agent_graph, method=deletion_method, threshold=deletion_threshold)
