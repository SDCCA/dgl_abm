"""This module contains functions which update edge weights.

Function(s):
- weight_update: Update edge weights based on agent wealth homophily
"""
import dgl.function as fn
import torch


def weight_update(agent_graph, device, homophily_parameter, characteristic_distance, 
                  truncation_weight):
    """Update function to calculate the weight of edges.
    
    Notes: Edge weight is based on the similarity in wealth of the connected nodes 
    according to the formula,

            weight = 1/(1 + e^(a*(m(x_i,x_j)-b))) 
    where:
        a = homophily parameter
        b = characteristic distance between the nodes in embedding space
        m(x_i, x_j) = difference in wealth between connected agents

    Weights falling below the numerical truncation value will be set at that value.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device (torch.device): Device on which to perform computations
        homophily_parameter (float): Parameter for homophily calculation, a
        characteristic_distance (float): Distance between nodes in embedding space, b
        truncation_weight (float): Minimum value for edge weights
    
    Returns:
        None
    """
    agent_graph.edata['weight'] = torch.rand(agent_graph.num_edges(),1).to(device)
    agent_graph.apply_edges(fn.u_sub_v('wealth','wealth','wealth_diff'))
    weights = 1./(1. + torch.exp(homophily_parameter*(torch.abs(
                            agent_graph.edata['wealth_diff'])-characteristic_distance)))
    finiteweights = torch.isfinite(weights)
    weights[~finiteweights] = 0.
    truncated_weights = torch.where( weights > truncation_weight, weights, 
                                    truncation_weight)
    agent_graph.edata['weight'] = truncated_weights
