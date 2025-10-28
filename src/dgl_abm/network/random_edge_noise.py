import dgl
import torch


def random_edge_noise(graph: dgl.DGLGraph, n_perturbations: int) -> None:
    """Adds random edge noise to the edges and potential edges of an agent network.

    Notes: This function adds noise to n graph edges by randomly sampling two nodes and assigning a random
    weight to an edge between them, removing and reinitializing edges that already exist and
    initializing edges for those that do not.

    Args:
        graph (DGLGraph): All agent node and edge data
        n_perturbations (int): Number of random edge perturbations to add to the graph
    """
    device = graph.device
    # Select neighbor node pairs randomly from graph and remove any duplicates and autoconnection suggestions
    node_pairs, _ = torch.sort(
        torch.stack(
            (
                torch.randint(0, graph.nodes().shape[0], (n_perturbations,)),
                torch.randint(0, graph.nodes().shape[0], (n_perturbations,)),
            ),
            dim=1,
        ),
        dim=1,
    )
    node_pairs = torch.unique(node_pairs, dim=0).to(device)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Delete existing edges between node pairs
    existing_connections = graph.has_edges_between(node_pairs[:, 0], node_pairs[:, 1])
    existing_forward = graph.edge_ids(node_pairs[:, 0][existing_connections], node_pairs[:, 1][existing_connections])
    existing_reverse = graph.edge_ids(node_pairs[:, 1][existing_connections], node_pairs[:, 0][existing_connections])
    graph.remove_edges(torch.cat((existing_forward, existing_reverse)))

    # Assign random weights to new node_pair edges
    random_weights = torch.rand(node_pairs.size(0), device=device).to(graph.edata["weight"].dtype)
    graph.add_edges(node_pairs[:, 0], node_pairs[:, 1], {"weight": random_weights})
    graph.add_edges(node_pairs[:, 1], node_pairs[:, 0], {"weight": random_weights})
