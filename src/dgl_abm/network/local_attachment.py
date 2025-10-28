"""This module contains a local attachment function for homophily-based edge formation.

Function:
- local_attachment_homophily: Attempts triad closure with a new edge
"""

import logging
from collections.abc import Callable
import dgl
import torch

logger = logging.getLogger(__name__)


def local_attachment(
    graph: dgl.DGLGraph, n_links: int, edge_property: None | str | dict[str, Callable] = None
) -> dgl.DGLGraph:
    """Attempt to complete triads with a new edge.

    Notes:
        This function may form "friends of friends" links between two agents connected
        by a common neighbor by randomly selecting 2 neighbors of nodes randomly
        selected from a pool of agents with 2 or more neighbors.
        n_links represents attempted links; sampling of connecting nodes is
        performed with replacement; if a connecting agent's sampled neighbors are
        already connected, a new link is not formed. A potentially connecting pair
        of neighbors is only considered once per call.


    Args:
        graph (DGLGraph): All agent node and edge data
        n_links (int): Number of new links to attempt
        edge_property (None|dict[str, Callable]|str): Dictionary with key(s) stating edge
            property for new edges and values as the corresponding function for populating
            the edge property (in case of string defaults to ones)

    Returns:
        graph (DGLGraph): Updated graph with new edges
    """
    device = graph.device

    # check type capacity
    if graph.idtype == torch.int32 and graph.number_of_edges() + n_links >= torch.iinfo(torch.int32).max:
        graph = graph.long()

    # preselect based on adjacency matrix for 2 or more neighbors
    candidates = graph.adj().sum(dim=1) > 1
    if torch.sum(candidates) == 0:
        logger.info("There are no agents with two or more neighbors. No local attachment can occur.")
        return graph

    # Select bridge/connecting nodes randomly from candidates (with replacement)
    connecting_nodes = torch.nonzero(candidates, as_tuple=True)[0][torch.randint(0, torch.sum(candidates), (n_links,))]

    # Sample 2 neighbors of bridge/connecting nodes and remove any duplicates or
    # autopairs
    sample = dgl.sampling.sample_neighbors(graph, connecting_nodes, 2, replace=False, edge_dir="out")
    node_pairs, _ = torch.sort(sample.edges(order="eid")[1].view(-1, 2), dim=1)
    node_pairs = torch.unique(node_pairs, dim=0)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Extract node pairs and exclude existing edges
    existing_connections = graph.has_edges_between(node_pairs[:, 0], node_pairs[:, 1])
    even_indices_tensor = node_pairs[:, 0][~existing_connections]
    odd_indices_tensor = node_pairs[:, 1][~existing_connections]

    # Add new edges to the original graph
    if isinstance(edge_property, str):
        edge_data = {edge_property: torch.ones_like(even_indices_tensor).to(graph.edata[edge_property].dtype)}

    elif isinstance(edge_property, dict):
        edge_data = {}
        for key, function in edge_property.items():
            edge_data[key] = function(even_indices_tensor, odd_indices_tensor).to(graph.edata[key].dtype)
    elif edge_property is None:
        edge_data = None
    else:
        message = "If provided, edge_propertymust be either a string or a dictionary of string:function pairs."
        raise TypeError(message)
    if edge_data:
        graph.add_edges(even_indices_tensor, odd_indices_tensor, data=edge_data)
        graph.add_edges(odd_indices_tensor, even_indices_tensor, data=edge_data)
    else:
        graph.add_edges(even_indices_tensor, odd_indices_tensor)
        graph.add_edges(odd_indices_tensor, even_indices_tensor)
    return graph.to(device)


def local_attachment_homophily(
    graph: dgl.DGLGraph, n_links: int, attribute: str | dict, truncation_weight: float | None = None
) -> dgl.DGLGraph:
    """Attempt to complete triads with a new edge based on homophily.

    Notes:
        This function may form "friends of friends" links between two agents connected
        by a common neighbor by randomly selecting 2 neighbors of nodes randomly
        selected from a pool of agents with 2 or more neighbors. The potential
        homophily weight of the new edge is calculated and the edge is formed if a
        randomly generated number is less than the potential homophily edge weight.
        n_FoF_links represents attempted links; sampling of connecting nodes is
        performed with replacement; if a connecting agent's sampled neighbors are
        already connected, or the random number generated is greater than the potential
        homophily edge weight, a new link is not formed. A potentially connecting pair
        of neighbors is only considered once per call.


    Args:
        graph (DGLGraph): All agent node and edge data
        n_links (int): Number of new links to attempt
        attribute (str|dict): Name of agent attribute for homophily calculation
            or dictionary with key matching an agent attribute and values to overwrite
            'homophily_parameter','characteristic_distance', and/or truncation_weight
            with None or string literal to use default calculation/values or a
            custom float value

        homophily_parameter (float): Parameter for weight calculation
        characteristic_distance (float): Distance between nodes in embedding space
        truncation_weight (float): Minimum value for edge weights

    Returns:
        graph (DGLGraph): Updated graph with new edges
    """
    device = graph.device
    if isinstance(attribute, dict) and len(attribute) > 1:
        message = (
            "Multiple attributes for homophily calculation not yet supported. "
            "Development underway in DGL-PTM spatial branch"
        )
        raise NotImplementedError(message)
    if isinstance(attribute, dict) and len(attribute) == 1:
        attribute_name = next(iter(attribute.keys()))
        homophily_parameter = attribute[attribute_name].get("homophily_parameter", 1.0)
        characteristic_distance = attribute[attribute_name].get(
            "characteristic_distance", _estimate_expected_distance(graph, attribute_name)
        )

    elif isinstance(attribute, str):
        attribute_name = attribute
        homophily_parameter = 1.0
        characteristic_distance = _estimate_expected_distance(graph, attribute_name)

    if attribute_name not in graph.ndata:
        attribute_mesage = f"Attribute '{attribute_name}' not found in graph node data."
        raise ValueError(attribute_mesage)

    if truncation_weight is None:
        truncation_weight = 1.0e-10

    # check type capacity
    if graph.idtype == torch.int32 and graph.number_of_edges() + n_links >= torch.iinfo(torch.int32).max:
        graph = graph.long()

    # preselect based on adjacency matrix for 2 or more neighbors
    candidates = graph.adj().sum(dim=1) > 1
    if torch.sum(candidates) == 0:
        logger.info("There are no agents with two or more neighbors. No local attachment can occur.")
        return graph

    # Select bridge/connecting nodes randomly from candidates (with replacement)
    connecting_nodes = torch.nonzero(candidates, as_tuple=True)[0][torch.randint(0, torch.sum(candidates), (n_links,))]

    # Sample 2 neighbors of bridge/connecting nodes and remove any duplicates or
    # autopairs
    sample = dgl.sampling.sample_neighbors(graph, connecting_nodes, 2, replace=False, edge_dir="out")
    node_pairs, _ = torch.sort(sample.edges(order="eid")[1].view(-1, 2), dim=1)
    node_pairs = torch.unique(node_pairs, dim=0)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Extract node pairs and exclude existing edges
    existing_connections = graph.has_edges_between(node_pairs[:, 0], node_pairs[:, 1])
    even_indices_tensor = node_pairs[:, 0][~existing_connections]
    odd_indices_tensor = node_pairs[:, 1][~existing_connections]
    prob_tensor = torch.rand(even_indices_tensor.size(0)).to(device)

    # Compare random number for each prospective link to projected homophily edge weight
    attribute_difference = (
        sample.ndata[attribute_name][even_indices_tensor] - sample.ndata[attribute_name][odd_indices_tensor]
    )
    potential_weights = 1.0 / (
        1.0 + torch.exp(homophily_parameter * (torch.abs(attribute_difference) - characteristic_distance))
    )
    finiteweights = torch.isfinite(potential_weights)
    potential_weights[~finiteweights] = 0.0
    potential_weights = torch.where(potential_weights > truncation_weight, potential_weights, truncation_weight)

    successful_links = potential_weights > prob_tensor

    # Add new edges to the original graph
    graph.add_edges(
        even_indices_tensor[successful_links],
        odd_indices_tensor[successful_links],
        data={"weight": potential_weights[successful_links]},
    )
    graph.add_edges(
        odd_indices_tensor[successful_links],
        even_indices_tensor[successful_links],
        data={"weight": potential_weights[successful_links]},
    )

    return graph.to(device)


def _estimate_expected_distance(graph: dgl.DGLGraph, attribute: str, calculation_limit: int = 10000) -> float:
    """Calculate/estimate the expectation for the distance between nodes in an attribute space.

    Args:
        graph (DGLGraph): All agent node and edge data
        attribute (str): Name of agent attribute for homophily calculation
        calculation_limit (int): Maximum number of node pairs to consider in calculation

    Returns:
        expected_distance (float): Expected distance between nodes (by default, estimated from
            sample if more than 10,000 nodes)
    """
    if graph.num_nodes() * (graph.num_nodes() - 1) <= calculation_limit:
        distances = torch.abs(graph.ndata[attribute].view(-1, 1) - graph.ndata[attribute].view(1, -1)).float()
        return torch.mean(torch.triu(distances, diagonal=1)).item()

    idx1 = torch.randint(0, graph.num_nodes(), (calculation_limit,))
    idx2 = torch.randint(0, graph.num_nodes(), (calculation_limit,))
    distances = torch.abs(graph.ndata[attribute][idx1] - graph.ndata[attribute][idx2]).float()
    distances = distances[idx1 != idx2]
    if distances.numel() == 0:
        message = (
            "All sampled pairs were self pairs. The expected distance is "
            "estimated at 0.0 for this graph. Try raising the calculation limit "
            "or providing a custom characteristic distance if this is unanticipated."
        )
        logger.warning(message)
        return 0.0
    return torch.mean(distances).item()
