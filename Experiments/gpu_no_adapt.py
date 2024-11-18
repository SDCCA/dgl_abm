import sys
sys.path.append('../dgl_ptm')
import dgl_ptm
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import argparse

def main(args):

    model = dgl_ptm.PovertyTrapModel(model_identifier=f'no_adapt_{args.seed}', root_path=args.root_path)

    model.set_model_parameters(**{'number_agents': args.agents, 
    'seed':args.seed,
    'sigma_dist': {'type':'uniform','parameters':[0.05,1.94],'round':True,'decimals':1},
    'a_theta_dist': {'type':'uniform','parameters':[0.1,1],'round':False,'decimals':None},
    'sensitivity_dist':{'type':'uniform','parameters':[0.0,1],'round':False,'decimals':None},
    'capital_dist': {'type':'uniform','parameters':[0.1,10.],'round':False,'decimals':None}, 
    'alpha_dist': {'type':'normal','parameters':[1.08,0.074],'round':False,'decimals':None},
    'lambda_dist': {'type':'uniform','parameters':[0.05,0.94],'round':True,'decimals':1},
    'initial_graph_type': 'barabasi-albert',
    'initial_graph_args': {'seed': args.seed, 'new_node_edges':5},
    'device': 'cuda',
    'step_target': args.steps,
    'steering_parameters':{'root_path': args.root_path,
                           'npath':'./agent_data.zarr',
                            'epath':'./edge_data', 
                            'ndata':[['degree','i_a','income','net_trade', 'tech_index','theta', 'wealth', 'wealth_consumption','weighted_degree'],['initial_only',['alpha','lambda','sigma','sensitivity']]],
                            'edata':None,
                            'mode':'w',
                            'capital_method':'past_shock',
                            'trade_method':'weighted_transfer',
                            'income_method':'income_generation',
                            'tech_gamma': torch.tensor([0.3,0.35,0.45]),
                            'tech_cost': torch.tensor([0,0.15,0.65]),
                            'consume_method':'past_shock_bellman_consumption_no_adapt',
                            'nn_path': "/nn_data/cons_PudgeSixLayer_2048/1105_215329/model_best.pth",
                            'adapt_m':torch.tensor([0]),
                            'adapt_cost':torch.tensor([0]),
                            'depreciation': 0.08,
                            'discount': 0.95,
                            'global_theta_dist': {'type':'beta','parameters':[4.13,0.07],'round':False,'decimals':None},
                            'del_method':'size',
                            'del_threshold':'balance',
                            'noise_ratio': 0.05,
                            'local_ratio': 0.25,
                            'homophily_parameter':1,
                            'characteristic_distance':3.33, 
                            'truncation_weight':1.0e-10,
                            'step_type':'ptm'}})

    model.initialize_model()
    print(f"Model successfully initialized \n Theta:{model.config.steering_parameters.global_theta}")
    model.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify model parameters")
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--root_path', default=os.getcwd(),type=str, required=False, help='Root path for output data')
    parser.add_argument('--agents', type=int, default=1000000, help='Number of agents')
    parser.add_argument('--steps', type=int, default=50, help='Number of timesteps')

    args = parser.parse_args()
    main(args)