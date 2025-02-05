#This code generates data for two figures with four subplots of a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os


#The root path:
#output_path= "/Volumes/PTM_data/PTMOutput/Disruption"
output_path= "output/Disruption"
#The data folder paths:
arrangement_paths = ["disrupt_6", "disrupt_7", "disrupt_8"]
arrangement_labels = ["default_"]


completed_seeds=[15796,861,76821,54887,6266,
                82387,37195,87499,44132,60264,
                16024,41091,67222,64821,770,
                59736,62956,64926,67970,5312,
                83105,53708,85306,28694,71933]           

#The target data:
graph_test_data=False

#ts_target=[25,50,75]
#target_columns=False
ts_target=[50]
target_columns=["i_a","wealth"]
full_initial_data=True
accumulations=True





def zarr_group_to_df(zarr_group, time_step=":", target_columns=False):
    """
    Convert a zarr group to a pandas dataframe.
    """

    df = pd.DataFrame()
    for array_name in zarr_group.array_keys():
        if target_columns is False or array_name in target_columns:
            zarr_array = zarr_group[array_name][:,time_step]
            df[array_name] = zarr_array.flatten()
            if array_name == "i_a" and accumulations!=False:
                zarr_array = zarr_group[array_name][:,0:time_step]
                df[f"{array_name}_accumulated"] = np.sum(zarr_array, axis=1)
            if array_name in ["theta","degree","wealth"] and full_initial_data!=False:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

# Create a dataframe from all available seeds for a model arrangement
# a target timestep specified above is used as a filter.

data_frames = {
    'disrupt_6': pd.DataFrame(),
    'disrupt_7': pd.DataFrame(),
    'disrupt_8': pd.DataFrame()
}

for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:
            if folder_name.startswith(arrangement_labels[0]):
                seed=int(folder_name.split('_')[-1])
                if seed in completed_seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in ts_target:
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns=target_columns)
                        working_df['seed']=seed
                        working_df['time_step']=ts
                        working_df["AgentID"] = working_df.index
                        data_frames[arrangement_path] = pd.concat([data_frames[arrangement_path], working_df], ignore_index=True)

dfs = list(data_frames.values())
labels = list(data_frames.keys())
print(data_frames["disrupt_6"].head())

disrupt_7_i_a = pd.DataFrame()
arrangement_paths = ["disrupt_7"]
for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:            
            if folder_name.startswith(arrangement_labels[0]):
                seed=int(folder_name.split('_')[-1])
                if seed in completed_seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in range(50):
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns="i_a")
                        working_df['seed']=seed
                        working_df['time_step']=ts
                        working_df['AgentID'] = working_df.index
                        disrupt_7_i_a = pd.concat([disrupt_7_i_a, working_df], ignore_index=True)

i_a_dfs = list(data_frames.values())
i_a_labels = list(data_frames.keys())
print(disrupt_7_i_a.head())

# Switching plot 


## upgrades and reductions
disrupt_7_i_a["i_a_previous"] = disrupt_7_i_a.groupby(["seed","AgentID"])["i_a"].shift(1)
disrupt_7_i_a["upgrades"] = (disrupt_7_i_a["i_a_previous"] < disrupt_7_i_a["i_a"]).astype(int)
disrupt_7_i_a["reductions"] = (disrupt_7_i_a["i_a_previous"] > disrupt_7_i_a["i_a"]).astype(int)
switch_df=pd.DataFrame({'time_step':range(50)})
switch_df['upgrades'] = disrupt_7_i_a[disrupt_7_i_a["time_step"]>0].groupby("time_step")["upgrades"].sum()/disrupt_7_i_a[disrupt_7_i_a["time_step"]>0].groupby("time_step").size().tolist()
switch_df["reductions"] = disrupt_7_i_a[disrupt_7_i_a["time_step"]>0].groupby("time_step")["reductions"].sum()/disrupt_7_i_a[disrupt_7_i_a["time_step"]>0].groupby("time_step").size().tolist()
switch_df.to_csv("disrupt_7_switches.csv", index=False)


def save_boxplot_data(total_investment, delta_k, filename, bin_size=5):
    max_investment = int(np.max(total_investment))
    bins = np.arange(0, max_investment + bin_size, bin_size)
    
    all_data = []
    for i in range(len(bins) - 1):
        bin_mask = (total_investment >= bins[i]) & (total_investment < bins[i + 1])
        bin_data = delta_k[bin_mask]
        if len(bin_data) > 0:
            min_val = np.min(bin_data)
            q1 = np.percentile(bin_data, 25)
            median = np.median(bin_data)
            q3 = np.percentile(bin_data, 75)
            max_val = np.max(bin_data)
            all_data.append({
                "bin_start": bins[i],
                "bin_end": bins[i + 1],
                "min": min_val,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max_val
            })
    boxplot_data = pd.DataFrame(all_data)
    boxplot_data.to_csv(filename, index=False)
    

# boxplot data for capital at final time step vs. total investment in adaptation


total_investment = data_frames['disrupt_7']["i_a_accumulated"]
delta_k = data_frames['disrupt_7']["wealth"]-data_frames['disrupt_7']["wealth_initial"]
save_boxplot_data(total_investment, delta_k, 'boxplot_bin_size_5.csv')





def save_boxplot_comparison(dfs, labels, filename, bin_size=2):
    all_data = []
    for df, label in zip(dfs, labels):
        df["delta_k"] = df["wealth"] - df["wealth_initial"]
        i_a = df["i_a_accumulated"]
        delta_k = df["delta_k"]
        
        bins = np.arange(min(i_a), max(i_a) + bin_size, bin_size)
        for i in range(len(bins) - 1):
            bin_mask = (i_a >= bins[i]) & (i_a < bins[i + 1])
            bin_data = delta_k[bin_mask]
            if len(bin_data) > 0:
                min_val = np.min(bin_data)
                q1 = np.percentile(bin_data, 25)
                median = np.median(bin_data)
                q3 = np.percentile(bin_data, 75)
                max_val = np.max(bin_data)
                all_data.append({
                    "bin_start": bins[i],
                    "bin_end": bins[i + 1],
                    "min": min_val,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": max_val,
                    "label": label
                })
    
    comparison_data = pd.DataFrame(all_data)
    comparison_data.to_csv(filename, index=False)

# Example usage
save_boxplot_comparison(dfs,['Disruption 0.6', 'Disruption 0.7', 'Disruption 0.8'], 'disruption_comparison_boxplot.csv')