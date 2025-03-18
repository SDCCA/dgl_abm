"""This module provides functions for agent consumption behavior.

Functions:
- wealth_consumption: Directs arguments to method-specific internal functions
- _pseudo_wealth_consumption: Sets consumption to a simple ratio of 1/3 of wealth
- _bellman_wealth_consumption: Previously used iterative Bellman equation solver
- _nn_bellman_wealth_consumption: Uses neural network to estimate consumption and i_a
- _nn_bellman_past_shock_consumption: Uses neural network to estimate 
    consumption and i_a with a second version of the Bellman equation
- _nn_bellman_past_shock_consumption_no_adapt: Neural network to estimate consumption
    with a second version of the Bellman equation
"""
import torch

from dgl_ptm.util.utils import load_consumption_model, scale_input


def wealth_consumption(model_graph, model_params, timestep=None, device=None, 
                       method='pseudo_consumption'):
    """Calculate agent consumption using one of multiple methods available.

    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time
        device (torch.device): Device on which to perform computations
        method (str): Method used for deciding consumption

    Returns:
        None
    """
    if method in ['default','pseudo','pseudo_consumption']:
        _pseudo_wealth_consumption(model_graph)
    elif method == 'calc_bellman_consumption':
        _bellman_wealth_consumption(model_graph,model_params)
    elif method == 'estimated_bellman_consumption':
        _nn_bellman_wealth_consumption(model_graph,model_params,device)
    elif method == 'past_shock_bellman_consumption':
        _nn_bellman_past_shock_consumption(model_graph,model_params, timestep, device)
    elif method == 'past_shock_bellman_consumption_no_adapt':
        _nn_bellman_past_shock_consumption_no_adapt(model_graph,model_params, timestep, 
                                                    device)
    else:
        raise NotImplementedError(f"Consumption method, {method}, is unavailable.")

def _pseudo_wealth_consumption(model_graph):
    """Set consumption to simple ratio of 1/3 of wealth.

    Arg:
        model_graph (DGLGraph): All agent node and edge data
    
    Returns:
        None
    """
    model_graph.ndata['wealth_consumption'] = model_graph.ndata['wealth']*1./3.

def _bellman_wealth_consumption(model_graph,model_params):
    """Approximate consumption behavior based on iterative Bellman equation output.
    
    Notes: This method is no longer supported.

    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration/initialization

    Returns:
        None
    """
    raise NotImplementedError("Iterative Bellman consumption approximation is no "
                              "longer supported. Please specify a different method for"
                              " consume_method in the config file.")

def  _nn_bellman_wealth_consumption(model_graph,model_params, device):
    """Estimate consumption and i_a using a pytorch neural network.
    
    Notes: Currently only works with a single model for all agents 
        under the four input two output configuration trained on iterative
        Bellman equation output.
        The entire surrogate model must be saved at nn_path and the architecture of the 
        model specified in nn_arch.py.

    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration and 
            initialization
        device (torch.device): Device on which to perform computations

    Returns:
        None
    """
    if model_params['nn_path'] is None:
        print("No consumption model path provided!")
    
    #load model  
    
    estimator,cons_scale,i_a_scale = load_consumption_model(model_params['nn_path'],
                                                            device)  

    estimator.to(device)
    estimator.eval()

    input = torch.cat((model_graph.ndata['alpha'].unsqueeze(1), 
                       model_graph.ndata['wealth'].unsqueeze(1), 
                       model_graph.ndata['sigma'].unsqueeze(1), 
                       model_graph.ndata['theta'].unsqueeze(1)), dim=1) 
    
    #forward pass to get predictions
    with torch.no_grad():

        pred=estimator(input)
    
    model_graph.ndata['m'], model_graph.ndata['i_a'] =(model_graph.ndata['a_table'][(
                torch.arange(model_graph.ndata['a_table'].size(0))),:,
                torch.argmin(torch.abs(pred[:, 0].unsqueeze(1)*i_a_scale - 
                model_graph.ndata['a_table'][:,1,:]), dim=1)].unbind(dim=1))
    
    #Clean Consumption
    model_graph.ndata['wealth_consumption']=(pred[:,1]*cons_scale).clamp_(min=0)

    # Check for violations
    # A violation occurs when (depreciated k + income - consumption - i_a) <= 0
    violation = ((1-model_params['depreciation']) * model_graph.ndata['wealth'] +
                model_graph.ndata['income']-model_graph.ndata['wealth_consumption'] -
                model_graph.ndata['i_a']<=0)


    if torch.sum(violation)!=0:
        # Violation type 1: i_a exceeds depreciated k + income
        violation_i_a = ((1-model_params['depreciation'])*model_graph.ndata['wealth'] +
                                model_graph.ndata['income']-model_graph.ndata['i_a']<=0)
        # Violation type 2: consumption exceeds k
        # violation_consumption = (model_graph.ndata['wealth'] -
        #                          model_graph.ndata['wealth_consumption']<=0)

        print(f"Agents in violation: {torch.sum(violation)}")
        print(f"...because of i_a: {torch.sum(violation_i_a)}")

        # setting i_a to 0 for type 1 violations
        model_graph.ndata['m'][torch.nonzero(violation_i_a, as_tuple=False)]=0
        model_graph.ndata['i_a'][torch.nonzero(violation_i_a, as_tuple=False)]=0

        # redetermine violations with updated i_a
        violation = ((1-model_params['depreciation']) * model_graph.ndata['wealth'] +
                model_graph.ndata['income'] - model_graph.ndata['wealth_consumption'] -
                model_graph.ndata['i_a']<=0)
        violation_i_a = ((1-model_params['depreciation']) * 
                         model_graph.ndata['wealth'] + model_graph.ndata['income'] - 
                         model_graph.ndata['i_a']<=0)

        model_graph.ndata['wealth_consumption'][torch.nonzero(violation, 
                                                              as_tuple = False)] = (
                            (1-model_params['depreciation']) * 
                            model_graph.ndata['wealth'][torch.nonzero(violation, 
                                                                as_tuple=False)] +
                            model_graph.ndata['income'][torch.nonzero(violation, 
                                                                as_tuple=False)] -
                            model_graph.ndata['i_a'][torch.nonzero(violation, 
                                                                as_tuple=False)]) * 0.99


    violation = ((1-model_params['depreciation'])*model_graph.ndata['wealth'] + 
            model_graph.ndata['income'] - model_graph.ndata['wealth_consumption'] - 
            model_graph.ndata['i_a']<=0)

    if torch.sum(violation)!=0:

        print(f"Something has gone terribly wrong! Still {torch.sum(violation)}" 
             " violations.")

def  _nn_bellman_past_shock_consumption(model_graph,model_params, timestep, device):
    """Estimate consumption and i_a using a pytorch neural network.

    Notes: In this version, k_t+1= model_graph.ndata['income'] + 
        (global_θ + m * (1-global_θ)) + (1-𝛿) * k - c - i_a.
        It currently only works with a single model for all agents under the four 
        input, two output configuration trained on iterative Bellman equation results. 
        The entire surrogate model must be saved at nn_path and the architecture of the 
        model specified in nn_arch.py.

    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time
        device (torch.device): Device on which to perform computations

    Returns:
        None
    """
    if model_params['nn_path'] is None:
        print("No consumption model path provided!")
    
    # Load model  
    estimator,cons_scale, i_a_scale,input_scale = load_consumption_model(
                                                        model_params['nn_path'],device)  

    estimator.to(device)
    estimator.eval()
    input = torch.cat((model_graph.ndata['alpha'].unsqueeze(1), 
                       model_graph.ndata['wealth'].unsqueeze(1), 
                       model_graph.ndata['sigma'].unsqueeze(1), 
                       model_graph.ndata['theta'].unsqueeze(1)), dim=1)
    # Scale inputs as specified in nn config
    if {"alpha","Alpha"} & input_scale.keys():
        input[:,0]=scale_input(input[:,0], input_scale.get(
                        list({"alpha","Alpha"} & input_scale.keys())[0]),"alpha",False)
    if {"k","K"} & input_scale.keys():
        input[:,1]=scale_input(input[:,1], input_scale.get(
                        list({"k","K"} & input_scale.keys())[0]),"k",False)
    if {"sigma","Sigma"} & input_scale.keys():
        input[:,2]=scale_input(input[:,2], input_scale.get(
                        list({"sigma","Sigma"} & input_scale.keys())[0]),"sigma",False)
    if {"theta","Theta"} & input_scale.keys():
        input[:,3]=scale_input(input[:,3], input_scale.get(
                        list({"theta","Theta"} & input_scale.keys())[0]),"theta",False)
    
    # Forward pass to get predictions
    with torch.no_grad():

        pred=estimator(input)
    
    model_graph.ndata['m'],model_graph.ndata['i_a']=model_graph.ndata['a_table'][(
        torch.arange(model_graph.ndata['a_table'].size(0))),:,
        torch.argmin(torch.abs(pred[:, 0].unsqueeze(1) * i_a_scale - 
                        model_graph.ndata['a_table'][:,1,:]), dim=1)].unbind(dim=1)
    print(f"scaled i_a: {model_graph.ndata['i_a'][0:5]}")

    # Clean Consumption
    print(f"Setting {torch.sum(pred[:,1]*cons_scale<0)} negative consumption "
          f"predictions to zero,{torch.sum(pred[:,1]*cons_scale<-0.1)} were less "
          "than -0.1 .")
    model_graph.ndata['wealth_consumption']=(pred[:,1]*cons_scale).clamp_(min=0)
    print( f"Based on alpha: {model_graph.ndata['alpha'][0:5]}")
    print(f'Based on k: {model_graph.ndata["wealth"][0:5]}')
    print(f'Based on sigma: {model_graph.ndata["sigma"][0:5]}')
    print(f'Based on theta: {model_graph.ndata["theta"][0:5]}')
    print(f"Prediction {pred[0:5].tolist()}")
    print(f"Planned consumption { model_graph.ndata['wealth_consumption'][0:5]}")
    print(f"Planned consumption percentage "
          f"""{((model_graph.ndata['wealth_consumption'][0:5] /
              model_graph.ndata['wealth'][0:5]) * 100)}""")
    print(f"Planned i_a { model_graph.ndata['i_a'][0:5]}")

    # Check for violations
    # An equation violation occurs when (personally shocked, depreciated k + income - 
    # consumption - i_a) is less than or equal to 0
    equation_violation = ((model_graph.ndata['theta'] + model_graph.ndata['m'] * 
                        (1-model_graph.ndata['theta'])) * 
                        (1-model_params['depreciation']) *
                        model_graph.ndata['wealth'] + model_graph.ndata['income'] - 
                        model_graph.ndata['wealth_consumption'] - 
                        model_graph.ndata['i_a']<=0)
    # A violation occurs when actually shocked, depreciated k + income - consumption - 
    # i_a is less than or equal to 0
    global_θ = model_params['global_theta'][timestep]
    violation = ((global_θ + model_graph.ndata['m'] * (1-global_θ)) *
                (1-model_params['depreciation']) * model_graph.ndata['wealth'] + 
                model_graph.ndata['income'] - model_graph.ndata['wealth_consumption'] - 
                model_graph.ndata['i_a']<=0)

    print(f"Agents in violation: {torch.sum(violation)}")

    if torch.sum(violation)!=0:
        # Violation type 1: i_a exceeds depreciated k + income
        violation_i_a = ((global_θ + model_graph.ndata['m'] * (1-global_θ)) * 
                (1-model_params['depreciation']) * model_graph.ndata['wealth'] + 
                model_graph.ndata['income'] - model_graph.ndata['i_a']<=0)

        print(f"Agents in equation violation: {torch.sum(equation_violation)}")
        print(f"Agents in personal/actual violation: {torch.sum(violation)}")
        print(f"...because of i_a: {torch.sum(violation_i_a)}")

        # setting i_a to 0 for type 1 violations
        model_graph.ndata['m'][torch.nonzero(violation_i_a, as_tuple=False)]=0
        model_graph.ndata['i_a'][torch.nonzero(violation_i_a, as_tuple=False)]=0

        # redetermine violations with updated i_a
        violation = ((global_θ + model_graph.ndata['m'] * (1-global_θ)) * 
                    (1-model_params['depreciation'])*model_graph.ndata['wealth'] + 
                    model_graph.ndata['income'] - 
                    model_graph.ndata['wealth_consumption'] - 
                    model_graph.ndata['i_a']<=0)
        violation_i_a = ((global_θ + model_graph.ndata['m'] * (1-global_θ)) * 
                    (1-model_params['depreciation']) * model_graph.ndata['wealth'] + 
                    model_graph.ndata['income'] - model_graph.ndata['i_a']<=0)

        model_graph.ndata['wealth_consumption'][torch.nonzero(violation, 
                                                              as_tuple=False)] = (
            (global_θ + model_graph.ndata['m'][torch.nonzero(violation, 
                                                             as_tuple=False)] * 
            (1-global_θ)) * (1-model_params['depreciation']) * 
            model_graph.ndata['wealth'][torch.nonzero(violation, as_tuple=False)] + 
            model_graph.ndata['income'][torch.nonzero(violation, as_tuple=False)] - 
            model_graph.ndata['i_a'][torch.nonzero(violation, as_tuple=False)]) * 0.99


    violation = ((global_θ + model_graph.ndata['m'] * (1-global_θ)) * 
                (1-model_params['depreciation'])*model_graph.ndata['wealth'] + 
                model_graph.ndata['income'] - model_graph.ndata['wealth_consumption'] - 
                model_graph.ndata['i_a']<=0)

    if torch.sum(violation)!=0:

        print(f"Something has gone terribly wrong! Still "
              f"{torch.sum(violation)} violations.")


def  _nn_bellman_past_shock_consumption_no_adapt(model_graph,model_params, 
                                                 timestep, device):
    """Estimate consumption using a pytorch neural network.

    k_{t+1} = model_graph.ndata['income'] + (global_θ) + (1-𝛿) * k - c

    Notes:
        This version was developed for special use with the no adaptation experiment.
        It currently only works with a single model for all agents under the four 
        input, one output configuration trained on iterative Bellman equation output.
        The entire surrogate model must be saved at nn_path and the architecture of the 
        model specified in nn_arch.py.

    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): Parameters specified in model configuration and 
            initialization
        timestep (int): Current model time
        device (torch.device): Device on which to perform computations

    Returns:
        None
    """
    if model_params['nn_path'] is None:
        print("No consumption model path provided!")
    
    # Load model  
    
    estimator,cons_scale, i_a_scale,input_scale = load_consumption_model(
                                                        model_params['nn_path'],device)  

    estimator.to(device)
    estimator.eval()
    
    input = torch.cat((model_graph.ndata['alpha'].unsqueeze(1), 
                       model_graph.ndata['wealth'].unsqueeze(1), 
                       model_graph.ndata['sigma'].unsqueeze(1), 
                       model_graph.ndata['theta'].unsqueeze(1)), dim=1)
    # Scale inputs as specified in nn config
    if {"alpha","Alpha"} & input_scale.keys():
        input[:,0]=scale_input(input[:,0], input_scale.get(list(
                    {"alpha","Alpha"} & input_scale.keys())[0]),"alpha",False)
    if {"k","K"} & input_scale.keys():
        input[:,1]=scale_input(input[:,1], input_scale.get(list(
                    {"k","K"} & input_scale.keys())[0]),"k",False)
    if {"sigma","Sigma"} & input_scale.keys():
        input[:,2]=scale_input(input[:,2], input_scale.get(list(
                    {"sigma","Sigma"} & input_scale.keys())[0]),"sigma",False)
    if {"theta","Theta"} & input_scale.keys():
        input[:,3]=scale_input(input[:,3], input_scale.get(list(
                    {"theta","Theta"} & input_scale.keys())[0]), "theta",False)
    
    # Forward pass to get predictions
    with torch.no_grad():

        pred=estimator(input)
    
    #model_graph.ndata['m'],model_graph.ndata['i_a'] are initialized as zeros
    # print("Cleaning output and checking for violations")

    # Clean Consumption
    print(f"Setting {torch.sum(pred[:,1]*cons_scale<0)} negative consumption "
          "predictions to zero,{torch.sum(pred[:,1]*cons_scale<-0.1)} were less "
          "than -0.1 .")
    model_graph.ndata['wealth_consumption']=(pred[:,0]*cons_scale).clamp_(min=0)

    # Check for violations
    # An equation violation occurs when personally shocked, 
    # depreciated k + income - consumption - i_a is less than or equal to 0
    equation_violation = ((model_graph.ndata['theta']) * 
                          (1-model_params['depreciation']) * 
                          model_graph.ndata['wealth'] + model_graph.ndata['income'] - 
                          model_graph.ndata['wealth_consumption'] <=0)
    # A violation occurs when actually shocked, depreciated k + income - consumption - 
    # i_a is less than or equal to 0
    global_θ = model_params['global_theta'][timestep]
    violation = ((global_θ)*(1-model_params['depreciation']) * 
                    model_graph.ndata['wealth']  + model_graph.ndata['income'] - 
                    model_graph.ndata['wealth_consumption'] <=0)



    if torch.sum(violation)!=0:
        # Violation type 1: i_a exceeds depreciated k + income
        
        # (Not possible for i_a = 0)

        # Violation type 2: consumption exceeds k
        # violation_consumption = model_graph.ndata['wealth'] -
        # model_graph.ndata['wealth_consumption']<=0

        print(f"Agents in equation violation: {torch.sum(equation_violation)}")
        print(f"Agents in personal/actual violation: {torch.sum(violation)}")

        # redetermine violations with updated i_a
        violation = ((global_θ ) * (1-model_params['depreciation']) * 
                     model_graph.ndata['wealth'] + model_graph.ndata['income'] - 
                     model_graph.ndata['wealth_consumption'] <=0)

        model_graph.ndata['wealth_consumption'][torch.nonzero(
            violation, as_tuple=False)] = (global_θ *(1-model_params['depreciation']) * 
            model_graph.ndata['wealth'][torch.nonzero(violation, as_tuple=False)] + 
            model_graph.ndata['income'][torch.nonzero(violation, as_tuple=False)])*0.99


    violation = ((global_θ ) * (1-model_params['depreciation']) * 
                model_graph.ndata['wealth']  + model_graph.ndata['income'] - 
                model_graph.ndata['wealth_consumption'] <=0)

    if torch.sum(violation)!=0:

        print(f"Something has gone terribly wrong! Still "
              f"{torch.sum(violation)} violations.")




