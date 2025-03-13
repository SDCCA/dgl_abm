"""This module contains functions which establish new edges.

Function(s):
- global_attachment: Randomly forms edges between agents
"""
import dgl
from dgl import AddEdge, AddReverse


def global_attachment(agent_graph, device, ratio: float):
    """Randomly connect agents globally based on a ratio.

    Note: If an attempted connection already exists, it will not be added again.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device: Device on which to perform computations
        ratio (float): decimal ratio of number of new edges to add to total existing 
            edges in graph

    Returns:
        None
    
    Effects:
        Modifies agent_graph by introducing new edges
    """
    # Add edges based on ratio
    agent_graph = AddEdge(ratio=ratio)(agent_graph)

    # Add reverse edges
    agent_graph = AddReverse()(agent_graph)

    # Remove duplicate edges
    # dgl.to_simple works only on device=cpu hence we move the graph to cpu:
    agent_graph = dgl.to_simple(agent_graph.to('cpu'), return_counts='cnt')
    # move the graph back to user choice of device.
    # This is necessary for running on cuda or other hardware.
    agent_graph = agent_graph.to(device)
