"""This module contains a local attachment function for homophily-based edge formation.

Function:
- local_attachment_homophily: Attempts triad closure with a new edge
"""
import dgl
import torch


def local_attachment_homophily(graph,device,n_links, homophily_parameter = None, 
                            characteristic_distance = None, truncation_weight = None):
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
        device (torch.device): Device on which to perform computations
        n_links (int): Number of new links to attempt
        homophily_parameter (float): Parameter for weight calculation
        characteristic_distance (float): Distance between nodes in embedding space
        truncation_weight (float): Minimum value for edge weights

    Returns:
        None
    
    Effects:
        Updates graph with new edges
    """
    #preselect based on adjacency matrix for 2 or more neighbors
    candidates=graph.adj().sum(dim=1)>1
    if torch.sum(candidates)==0:
        print("There are no agents with two or more neighbors. No local attachment"
              " can occur.")
        return

    # Select bridge/connecting nodes randomly from candidates (with replacement)
    connecting_nodes=torch.nonzero(candidates, as_tuple=True)[0][torch.randint(
                                            0,torch.sum(candidates),(n_links,))]

    # Sample 2 neighbors of bridge/connecting nodes and remove any duplicates or 
    # autopairs
    sample = dgl.sampling.sample_neighbors(graph,connecting_nodes, 2 , replace= False, 
                                           edge_dir="out")
    node_pairs,_ = torch.sort(sample.edges(order='eid')[1].view(-1, 2), dim=1)
    node_pairs = torch.unique(node_pairs, dim=0)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Extract node pairs and exclude existing edges
    existing_connections = graph.has_edges_between(node_pairs[:,0], node_pairs[:,1])
    even_indices_tensor = node_pairs[:,0][~existing_connections]
    odd_indices_tensor = node_pairs[:,1][~existing_connections]
    prob_tensor = torch.rand(even_indices_tensor.size(0)).to(device)

    # Compare random number for each prospective link to projected homophily edge weight
    wealth_diff = (sample.ndata['wealth'][even_indices_tensor] - 
                                            sample.ndata['wealth'][odd_indices_tensor])
    potential_weights = 1./(1. + torch.exp(homophily_parameter * 
                                (torch.abs(wealth_diff) - characteristic_distance)))
    finiteweights = torch.isfinite(potential_weights)
    potential_weights[~finiteweights] = 0.
    potential_weights = torch.where(potential_weights > truncation_weight, 
                                                potential_weights, truncation_weight)

    successful_links = potential_weights > prob_tensor

    # Add new edges to the original graph
    graph.add_edges(even_indices_tensor[successful_links], 
                    odd_indices_tensor[successful_links], 
                    data={'weight': potential_weights[successful_links]})
    graph.add_edges(odd_indices_tensor[successful_links], 
                    even_indices_tensor[successful_links], 
                    data={'weight': potential_weights[successful_links]})





