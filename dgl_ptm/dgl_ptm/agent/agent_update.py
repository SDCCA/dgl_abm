"""This module contains (or directs arguments to) functions updating agent properties.

Functions:
- agent_update: Directs arguments to method-specific internal functions
- _pseudo_agent_update: Updates agent wealth based on income and consumption
- _agent_capital_update: Directs arguments to `capital_update` module
- _agent_theta_update: Updates agent perception of theta
- _agent_consumption_update: Directs arguments to `wealth_consumption` module
- _agent_income_update: Directs arguments to `income_generation` module
- _agent_degree_update: Updates agent degree
- _agent_weighted_degree_update: Updates agent weighted degree
"""
from dgl_ptm.agent.capital_update import capital_update
from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
from dgl_ptm.util.network_metrics import node_degree, node_weighted_degree


def agent_update(model_graph, model_params=None, device=None, 
                 timestep=None, method='pseudo'):
    """Update agent attributes according to specified method.
    
    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration and initialization
        device (torch.device): Device on which to perform computations
        timestep (int): Current model time
        method (str): Method for updating capital

    Returns:
        None
    """
    if method == 'capital':
        _agent_capital_update(model_graph, model_params, timestep)
    elif method == 'theta':
        _agent_theta_update(model_graph, model_params, timestep)
    elif method == 'consumption':
        _agent_consumption_update(model_graph, model_params, timestep, device)
    elif method == 'income':
        _agent_income_update(model_graph,model_params,device)
    elif method == 'degree':
        _agent_degree_update(model_graph)
    elif method == 'weighted_degree':
        _agent_weighted_degree_update(model_graph)
    elif method in ['default','pseudo']:
        _pseudo_agent_update(model_graph,model_params,device)
    else:
        raise NotImplementedError(f"Unrecognized agent update type {method} attempted "
                                  f"during time step implementation.")

def _pseudo_agent_update(model_graph,model_params,device): 
    """Update agent wealth based on income and consumption.
    
    Arg:
    model_graph (DGLGraph): All agent data

    Returns:
    None
    """
    capital_update(model_graph)


def _agent_capital_update(model_graph,model_params,timestep):
    """Update agent capital based on method specified in model parameters.

    Note: k_t+1 becomes the new k_t

    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time

    Returns:
        None
    """
    capital_update(model_graph, model_params, timestep, 
                   method=model_params['capital_method'])

    
def _agent_theta_update(model_graph,model_params,timestep):
    """Update agent perception of theta based on observation and sensitivity.

    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration /initialization
        timestep (int): Current model time

    Returns:
        None
    """
    global_θ =model_params['global_theta'][timestep]
    model_graph.ndata['theta'] = (model_graph.ndata['theta'] * 
                                  (1-model_graph.ndata['sensitivity']) + 
                                  global_θ * model_graph.ndata['sensitivity'])

def _agent_consumption_update(model_graph, model_params, timestep, device):
    """Update agent consumption based on method specified in model parameters.
    
    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time
        device (torch.device): Device on which to perform computations

    Returns:
        None
    """
    wealth_consumption(model_graph, model_params,timestep, device, 
                       method=model_params['consume_method'])

def _agent_income_update(model_graph, model_params, device):
    """Update agent income based on method specified in model parameters.
    
    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        device (torch.device): Device on which to perform computations
    
    Returns:
        None
    """
    income_generation(model_graph,device,model_params,
                      method=model_params['income_method'])

def _agent_degree_update(model_graph):
    """Update agent degree.
    
    Note: Both directions (in/out) are considered.
    
    Arg:
        model_graph (DGLGraph): All agent data

    Returns:
        None
    """
    model_graph.ndata['degree'] = node_degree(model_graph)

def _agent_weighted_degree_update(model_graph):
    """Update agent weighted degree.
    
    Note: both directions (in/out) are considered.
    
    Arg:
        model_graph (DGLGraph): All agent data

    Returns:
        None
    """
    model_graph.ndata['weighted_degree'] = node_weighted_degree(model_graph)



