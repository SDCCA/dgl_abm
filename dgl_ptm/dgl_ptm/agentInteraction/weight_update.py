import dgl.function as fn
import torch

def weight_update(agent_graph, device, homophily_parameter, characteristic_distance, truncation_weight):
    """
    Update function to calculate the weight of edges based on the wealth 
    of the connected nodes according to the formula:
            weight = 1/(1 + e^(a*(m(x_i,x_j)-b))) 
    where:
        a = homophily parameter
        b = characteristic distance between the nodes in embedding space
        m(x_i, x_j) = difference in wealth between connected agents

    weights falling below the numerical truncation value will be set at that value
    """
    agent_graph.edata['weight'] = torch.rand(agent_graph.num_edges(),1).to(device)
    agent_graph.apply_edges(fn.u_sub_v('wealth','wealth','wealth_diff'))
    weights = 1./(1. + torch.exp(homophily_parameter*(torch.abs(agent_graph.edata['wealth_diff'])-characteristic_distance)))
    finiteweights = torch.isfinite(weights)
    weights[~finiteweights] = 0.
    truncated_weights = torch.where( weights > truncation_weight, weights, truncation_weight)
    agent_graph.edata['weight'] = truncated_weights

def multi_property_weight_update(agent_graph, device, truncation_weight, properties={'wealth':{'keys':["wealth"],'homophily_parameter':2,'characteristic_distance':3.33},'position':{'keys':["x","y"],'homophily_parameter':2,'characteristic_distance':1}}):
    """
    Update function to calculate the weight of edges based on the properties 
    of the connected nodes according to the formula:
            weight = 1/n * Σ(1/(1 + e^(a_k*(m(x_i,x_j)-b_k)))) 
    where:
        k = property
        a = homophily parameter
        b = characteristic distance between the nodes in embedding space
        m(x_i, x_j) = distance/differece between connected agents


    weights falling below the numerical truncation value will be set at that value
    """
    agent_graph.edata['weight'] = torch.zeros(agent_graph.num_edges()).to(device)
    for property in properties:
        
        if len(properties[property]['keys']) == 1:
            agent_graph.apply_edges(fn.u_sub_v(properties[property]['keys'][0],properties[property]['keys'][0],f"{str(properties[property]['keys'][0])}_diff"))
            weights = 1./(1. + torch.exp(properties[property]['homophily_parameter']*(torch.abs(agent_graph.edata[f"{str(properties[property]['keys'][0])}_diff"])-properties[property]['characteristic_distance'])))/len(properties)
            finiteweights = torch.isfinite(weights)
            weights[~finiteweights] = 0.
            truncated_weights = torch.where( weights > truncation_weight, weights, truncation_weight)
            agent_graph.edata['weight'] += truncated_weights
        elif len(properties[property]['keys']) == 2:
            agent_graph.apply_edges(fn.u_sub_v(properties[property]['keys'][0],properties[property]['keys'][0],f"{str(properties[property]['keys'][0])}_diff"))
            agent_graph.apply_edges(fn.u_sub_v(properties[property]['keys'][1],properties[property]['keys'][1],f"{str(properties[property]['keys'][1])}_diff"))
            weights = 1./(1. + torch.exp(properties[property]['homophily_parameter']*(torch.sqrt((agent_graph.edata[f"{str(properties[property]['keys'][0])}_diff"])**2+(agent_graph.edata[f"{str(properties[property]['keys'][1])}_diff"])**2)-properties[property]['characteristic_distance'])))/len(properties)
            finiteweights = torch.isfinite(weights)
            weights[~finiteweights] = 0.
            truncated_weights = torch.where( weights > truncation_weight, weights, truncation_weight)
            agent_graph.edata['weight'] += truncated_weights
        else:
            raise ValueError("Only 1 or 2 dimensional distances are currently supported for homophily")