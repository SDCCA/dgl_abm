"""This module contains functions which establish new edges.

Function(s):
- global_attachment: Randomly forms edges between agents
"""

import math
import dgl
import torch
from dgl import AddEdge
from dgl import AddReverse


def global_attachment(graph: dgl.DGLGraph, ratio: float) -> dgl.DGLGraph:
    """Randomly connect agents globally based on a ratio.

    Note: If an attempted connection already exists, it will not be added again.

    Args:
        graph (DGLGraph): All agent node and edge data
        ratio (float): decimal ratio of number of new edges to add to total existing
            edges in graph

    Returns:
        graph (DGLGraph): Updated graph with new edges added
    """
    device = graph.device
    # Convert to long type if necessary
    if (
        graph.idtype == torch.int32
        and graph.number_of_edges() + math.ceil(ratio * graph.number_of_nodes()) >= torch.iinfo(torch.int32).max
    ):
        graph = graph.long()

    # Add edges based on ratio
    graph = AddEdge(ratio=ratio)(graph)

    # Add reverse edges
    graph = AddReverse()(graph)

    # Remove duplicate edges
    # dgl.to_simple works only on device=cpu hence we move the graph to cpu:
    # to_simple by default copies ndata but not edata, hence we need to copy edata explicitly.
    graph = dgl.to_simple(graph.to("cpu"), return_counts="cnt", copy_edata=True)
    # move the graph back to user choice of device.
    # This is necessary for running on cuda or other hardware.

    return graph.to(device)
