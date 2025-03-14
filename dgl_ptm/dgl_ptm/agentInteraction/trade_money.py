"""This module contains functions dictating agent resource exchange.

Functions:
- trade_money: Directs arguments to method-specific internal functions
- _weighted_transfer: Distributes wealth to all connected agents based on edge weights
- _singular_transfer: Distributes wealth to one randomly selected connected agent
"""
import dgl
import dgl.function as fn
import torch


def trade_money(agent_graph, device, method: str):
    """Exchanges money between connected agents according to specified method.
    
    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device: Device on which to perform computations
        method (str): Method for exchanging wealth

    Returns:
        None
            
    NOTE: Exchange is based on wealth (k) and savings propensity (lambda) and assumes 
        that the following properties are available already:
        k, lambda, w (for 'weighted_transfer'), zeros, ones, total neighbour count
    NOTE: All edges are bidirected with uniform weights 'w'

    TODO: Rename variables as per Thijs' updates on notebook
    """
    # Calculating disposable wealth
    print(f"k before:{agent_graph.ndata['wealth'][0:5]}")
    agent_graph.ndata['disposable_wealth'] = (agent_graph.ndata['lambda'] * 
        agent_graph.ndata['wealth']) # TODO: declare what lambda is
    
    # Transfer of wealth
    if method == 'weighted_transfer':
        _weighted_transfer(agent_graph, device)
    elif method == 'singular_transfer':
        _singular_transfer(agent_graph, device)
    elif method == 'no_transfer':
        pass
    else:
        raise NotImplementedError(f"Unrecognized exchange type {method} attempted "
                                  f"during time step implementation.")
    
def _weighted_transfer(agent_graph, device):
    """Transfer wealth from each agent (node) to every connected neighbour by weight.
    
    Note: Based on pre-defined edge weights stored as edge properties

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device: Device on which to perform computations
    
    Returns:
        None
    
    Effects:
        agent_graph.ndata['net_trade']: Updates node attribute 'net_trade' with sum 
            of capital transfered from connected agent nodes to self minus the 
            outflow of capital to connected agent nodes within the time-step
        agent_graph.ndata['wealth']: Updates node attribute 'wealth' with the
            sum of the previous wealth and net_trade
    """
    # Sum all incoming weights
    agent_graph.ndata['total_weight'] = torch.zeros(agent_graph.num_nodes()).to(device)
    agent_graph.update_all(fn.u_add_e('total_weight','weight','total_weight_msg'), 
                           fn.sum('total_weight_msg', 'total_weight'))

    # Calculating outgoing weight %s
    agent_graph.apply_edges(fn.e_div_u('weight','total_weight','percent_weight'))

    # Wealth transfer amount on each edge
    agent_graph.apply_edges(fn.e_mul_u('percent_weight','disposable_wealth',
                                       'trfr_wealth'))  
    # TODO: check what trfr_wealth actually is

    # Sum total incoming wealth
    agent_graph.update_all(fn.v_add_e('zeros','trfr_wealth','net_trade_msg'), 
                           fn.sum('net_trade_msg', 'net_trade'))
    print(f"Total In:{agent_graph.ndata['net_trade'][0:5]}")

    # Subtract outgoing wealth
    agent_graph.ndata['net_trade'] = (agent_graph.ndata['net_trade'] - 
                                                agent_graph.ndata['disposable_wealth'])
    print(f"Disposable Wealth:{agent_graph.ndata['disposable_wealth'][0:5]}")  
    print(f"Net Trade:{agent_graph.ndata['net_trade'][0:5]}")

    # Conduct exchange
    agent_graph.ndata['wealth'] = (agent_graph.ndata['wealth'] + 
                                        agent_graph.ndata['net_trade'])
    print(f"k after:{agent_graph.ndata['wealth'][0:5]}")

def _singular_transfer(agent_graph, device):
    """Transfer wealth from each agent (node) to one randomly selected neighbour.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device: Device on which to perform computations

    Returns:
        None
    
    Effects:
        agent_graph.ndata['net_trade']: Updates node attribute 'net_trade' with sum 
            of capital transfered from connected agent node(s) to self minus the 
            outflow of capital to the selected agent node within the time-step
        agent_graph.ndata['wealth']: Updates node attribute 'wealth' with the
            sum of the previous wealth and net_trade
    """
    # Subsample graph to one random edge per node
    graph_subset = dgl.sampling.sample_neighbors(agent_graph, agent_graph.nodes(), 1, 
                                                 edge_dir='out', copy_ndata = True, 
                                                 output_device = device)
    
    # Calculate incoming wealth for each agent in subgraph
    graph_subset.ndata['net_trade'] = torch.zeros(agent_graph.num_nodes()).to(device)
    graph_subset.update_all(fn.u_add_v('disposable_wealth','zeros','net_trade_msg'), 
                            fn.sum('net_trade_msg', 'net_trade'))
    
    # Update wealth delta in agent graph
    agent_graph.ndata['net_trade'] = graph_subset.ndata['net_trade']
    agent_graph.ndata['net_trade'] = (agent_graph.ndata['net_trade'] - 
                                            agent_graph.ndata['disposable_wealth'])
    
    # Conduct exchange
    agent_graph.ndata['wealth'] = (agent_graph.ndata['wealth'] + 
                                                    agent_graph.ndata['net_trade'])