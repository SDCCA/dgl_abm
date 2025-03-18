#This code generates plots with four subplots for a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(filename='output/equality_checker_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

#The root path:
#output_path= "/Volumes/PTM_data/PTMOutput"
output_path= "output"
#The data folder paths:
default_path= "default"
no_social_path= "no_social"
no_adaptation_path= "no_adapt"
null_path= "null"
completed_seeds=[15796,861,76821,54887,6266,
                82387,37195,87499,44132,60264,
                16024,41091,67222,64821,770,
                59736,62956,64926,67970,5312,
                83105,53708,85306,28694,71933]  

#The target timestep:
ts_target=50

graph_test_data=False


def zarr_group_to_df(zarr_group, time_step=":", target_columns=False):
    """
    Convert a zarr group to a pandas dataframe.
    """

    df = pd.DataFrame()
    for array_name in zarr_group.array_keys():
        if target_columns is False or array_name in target_columns:
            zarr_array = zarr_group[array_name][:,time_step]
            logging.info(f'{array_name, zarr_array.shape}')
            df[array_name] = zarr_array.flatten()
            if array_name == "i_a":
                zarr_array = zarr_group[array_name][:,0:time_step]
                df[f"{array_name}_accumulated"] = np.sum(zarr_array, axis=1)
            if array_name in ["theta","degree"]:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

for _,dirnames,_ in os.walk(os.path.join(output_path,default_path)):
    for folder_name in dirnames:
        if folder_name.startswith("default_"):
            seed=int(folder_name.split('_')[1])
            if seed in completed_seeds:
                agent_initial_d=zarr.open(os.path.join(output_path,default_path,folder_name,"agent_data_initial.zarr"), mode='r')
                agent_initial_d=zarr_group_to_df(agent_initial_d, time_step=0)
                agent_initial_ns= zarr.open(os.path.join(output_path,no_social_path,f"no_social_{seed}/agent_data_initial.zarr"), mode='r')
                agent_initial_ns=zarr_group_to_df(agent_initial_ns, time_step=0)
                agent_initial_na= zarr.open(os.path.join(output_path,no_adaptation_path,f"no_adapt_{seed}/agent_data_initial.zarr"), mode='r')
                agent_initial_na=zarr_group_to_df(agent_initial_na, time_step=0)
                agent_initial_n=zarr.open(os.path.join(output_path,null_path,f"null_{seed}/agent_data_initial.zarr"), mode='r')
                agent_initial_n=zarr_group_to_df(agent_initial_n, time_step=0)

#Check if the initial agents are identical for all models:
                logging.info(f"\nChecking equality for seed {seed}")
                if not agent_initial_d.equals(agent_initial_ns) and agent_initial_d.equals(agent_initial_na) and agent_initial_d.equals(agent_initial_n):
                    if not agent_initial_d.equals(agent_initial_ns):
                        logging.info("Initial agents are not identical for default and no social.")
                    if not agent_initial_d.equals(agent_initial_na):
                        logging.info("Initial agents are not identical for default and no adaptation.")
                    if not agent_initial_d.equals(agent_initial_n):
                        logging.info("Initial agents are not identical for default and null.")
                    if not agent_initial_ns.equals(agent_initial_na):
                        logging.info("Initial agents are not identical for no social and no adaptation.")
                    if not agent_initial_ns.equals(agent_initial_n):
                        logging.info("Initial agents are not identical for no social and null.")
                    if not agent_initial_na.equals(agent_initial_n):
                        logging.info("Initial agents are not identical for no adaptation and null.")
                else:
                    logging.info("Agents were initialized identically for all four arrangements.")
            