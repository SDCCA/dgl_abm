"""This module contains functions to initialize a network for agents in the model.

Function(s):
- network_creation: Generates the network between the initialized nodes
- barabasi_albert_graph: Generates a barabasi-albert graph
"""

import dgl
import networkx as nx
import torch


def network_creation(num_agents, method, **kwargs):
    """Create the network between the initialized nodes using edges from DGL.

    Args:
        num_agents (int): Number of agent nodes
        method (str): Network creation method
            barabasi_albert: 
                This method takes the following possible keyword arguments,
                seed: random seed for networkx barabasi_albert_graph function
                new_node_edges: number of edges to create for each new node
        kwargs (dict): keyword arguments to be supplied to the network creation method

    Return:
        agent_graph (DGLGraph): network of agent nodes resulting from the chosen method
    """
    if (method == 'barabasi-albert'):
        if 'seed' in kwargs.keys():
            seed  = kwargs['seed']
        else:
            seed = torch.initial_seed() 
        
        if 'new_node_edges' in kwargs.keys(): 
            new_node_edges = kwargs['new_node_edges']
        else:
            new_node_edges = 1 
        print(f"Using seed {seed} for network creation with {new_node_edges} "
              "edges requested.")
        agent_graph = barabasi_albert_graph(num_agents, new_node_edges, seed)
    else:
        raise NotImplementedError('Currently only barabasi-albert model implemented!')
    
    return agent_graph

def barabasi_albert_graph(num_agents, new_node_edges=1, seed=1):
    """Create a barabasi-albert graph.
    
    This function creates a network graph for user-defined
    number of agents using the barabasi albert model function 
    from networkx.

    Args:
        num_agents (int): number of agent nodes
        new_node_edges (int): number of edges to create for each new node
        seed (int): random seed for function

    Return:
        agent_graph (DGLGraph): network of agent nodes resulting from the chosen method
    """
    #Create graph using networkx function for barabasi albert graph 
    networkx_graph = nx.barabasi_albert_graph(n=num_agents, m=new_node_edges, seed=seed)
    barabasi_albert_coo = nx.to_scipy_sparse_array(networkx_graph,format='coo')
    
    #Return DGL graph from networkx graph
    return dgl.from_scipy(barabasi_albert_coo)
