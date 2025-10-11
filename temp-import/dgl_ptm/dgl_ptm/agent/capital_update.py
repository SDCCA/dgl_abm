"""This module provides functions for updating agent capital.

Functions:
- capital_update: Directs arguments to method-specific internal functions
- _pseudo_capital_update: Simple capital update based on income and consumption
- _agent_capital_update: Update agent capital including shock from current timestep
- _agent_capital_past_shock_update: Update agent capital including shock from 
    previous timestep
"""
def capital_update(model_graph, model_params=None, timestep=None, method='default'):
    """Update agent capital according to specified method.
    
    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time
        method (str): Method for updating capital

    Returns:
        None
    """
    if method in ['pseudo','default']:
        _pseudo_capital_update(model_graph)
    elif method == 'present_shock':
        _agent_capital_update(model_graph, model_params, timestep)
    elif method == 'past_shock':
        _agent_capital_past_shock_update(model_graph, model_params, timestep)
    else:
        raise NotImplementedError(f'Capital update method "{method}" is unavailable.')

def _pseudo_capital_update(model_graph):
    """Update wealth with result of stock capital + income - consumption.

    Arg:
        model_graph (DGLGraph): All agent data

    Returns:
        None
    """
    model_graph.ndata['wealth'] = (model_graph.ndata['wealth'] + 
                                   model_graph.ndata['income'] - 
                                   model_graph.ndata['wealth_consumption'])

def _agent_capital_update(model_graph,model_params,timestep):
    """Update agent capital.
    
    Note: Applies shock from previous timestep to entire stock of new capital

    Args:
        model_graph (DGLGraph): All agent data    
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time

    Returns:
        None
    """
    k,c,i_a,m = (model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],
    model_graph.ndata['i_a'],model_graph.ndata['m'])
    global_θ = model_params['global_theta'][timestep-1]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = ((global_θ + m * (1-global_θ)) * 
                                   (model_graph.ndata['income'] - c - i_a + (1-𝛿) * k))

def _agent_capital_past_shock_update(model_graph,model_params,timestep):
    """Update agent capital.
    
    Note: Applies shock from previous timestep to any capital carried over.

    Args:
        model_graph (DGLGraph): All agent data    
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time

    Returns:
        None
    """
    k,c,i_a,m = (model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],
                 model_graph.ndata['i_a'],model_graph.ndata['m'])
    global_θ = model_params['global_theta'][timestep-1]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = (model_graph.ndata['income'] + (global_θ + m * 
                                    (1-global_θ)) + (1-𝛿) * k - c - i_a)

