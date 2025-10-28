"""This module contains the functions to delete graph edges.

Function(s):
- link_deletion: Deletes edges between agents
- _select_edges: Identifies edges to delete according to specified deletion method
"""

import torch
from dgl import DGLGraph
from dgl_abm.util import matrix_utils


def link_deletion(agent_graph: DGLGraph, method: str, threshold: float) -> None:
    """Delete links between agents according to a specified deletion method.

    Note: In case of the "weighted" and "multinomial" methods, the probability of
        selecting an edge for deletion is (multiplied by) the inverse weight
        of that edge.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        method (str): deletion method among options:
            "probability": each edge selected independently with equal probability,
            "weighted": each edge selected independently with weighted probability,
            "size": fixed number of edges selected with equal probability, or
            "multinomial": fixed number of edges selected with weighted probability.
        threshold(float): Threshold for deleting an existing edge between two nodes.
            The interpretation of this threshold depends on the deletion method:
            "probability": the probability of deleting any edge,
            "weighted": the base probability of deleting any edge,
            "size": the number of edges to delete, or
            "multinomial": the number of edges to delete

    Returns:
        None

    Effects:
        Updates agent_graph with reduced edges based on 'method' and 'threshold'.
    """
    agent_graph.remove_edges(_select_edges(agent_graph, method=method, threshold=threshold))


def _select_edges(agent_graph: DGLGraph, method: str, threshold: float) -> torch.Tensor:
    """Identify edges to delete according to specified deletion method.

    Note: In case of the "weighted" and "multinomial" methods, the probability of
        selecting an edge for deletion is (multiplied by) the inverse weight of
        that edge.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        method (str): deletion method. Must be either:
            "probability": each edge selected idependently with equal probability,
            "weighted": each edge selected independently with weighted probability,
            "size": fixed number of edges selected with equal probability, or
            "multinomial": fixed number of edges selected with weighted probability
        threshold(float): Threshold for deleting an existing edge between two nodes
            The interpretation of this threshold depends on the deletion method:
            "probability": the probability for deleting any edge,
            "weighted": the base probability for deleting any edge,
            "size": the number of edges to delete, or
            "multinomial": the number of edges to delete

    Returns:
        torch.Tensor: Edge IDs to delete from agent_graph based on 'method'
                and 'threshold'

    Effects:
        Updates agent_graph with reduced edges based on 'method' and 'threshold'
    """
    upper_triangular = matrix_utils.sparse_matrix_to_upper_triangular(agent_graph.adj())

    if method == "probability":
        mask_edges = torch.rand(upper_triangular.val.size()[0]) < threshold
    elif method == "weighted":
        mask_edges = ((1.0 - agent_graph.edata["weight"]) * torch.rand(upper_triangular.val.size()[0])) < threshold
    elif method == "size":
        mask_edges = torch.zeros(upper_triangular.val.size()[0]).bool()
        random_order = torch.rand(upper_triangular.val.size()[0])
        _, random_edges = torch.sort(random_order)
        mask_edges[random_edges[0 : int(threshold)]] = True
    elif method == "multinomial":
        mask_edges = torch.zeros(upper_triangular.val.size()[0])
        eid = (1.0 - agent_graph.edata["weight"]).multinomial(threshold, replacement=False)
        mask_edges.scatter_(0, eid, 1.0)
    else:
        message = 'Currently only "probability" and "size" deletion methods are supported'
        raise NotImplementedError(message)
        mask_edges = torch.zeros(upper_triangular.val.size()[0])

    deletion_matrix_upper_tri = matrix_utils.apply_mask_to_sparse_matrix(upper_triangular, mask_edges)
    deletion_matrix = matrix_utils.upper_triangular_to_symmetrical(deletion_matrix_upper_tri)

    return agent_graph.edge_ids(deletion_matrix.row, deletion_matrix.col)
