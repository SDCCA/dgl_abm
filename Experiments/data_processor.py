#This code generates data for two figures with four subplots of a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed



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
            print(array_name, zarr_array.shape)
            df[array_name] = zarr_array.flatten()
            if array_name == "i_a":
                zarr_array = zarr_group[array_name][:,0:time_step]
                df[f"{array_name}_accumulated"] = np.sum(zarr_array, axis=1)
            if array_name in ["theta","degree","wealth"]:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

def calculate_mutual_info(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    mi = mutual_info_regression(X, y)
    return mi


# Create a dataframe from all available seeds for a model arrangement
# a target timestep specified above is used as a filter.
default_df=pd.DataFrame()
for _,dirnames,_ in os.walk(os.path.join(output_path,default_path)):
    for folder_name in dirnames:
        if folder_name.startswith("default_"):
            print(f"Processing{os.path.join(output_path,default_path,folder_name,'agent_data.zarr')}")
            seed=int(folder_name.split('_')[-1])
            if seed in completed_seeds:
                zarr_path = os.path.join(output_path,default_path,folder_name,'agent_data.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                working_df = zarr_group_to_df(zarr_array, time_step=ts_target)
                working_df['seed']=seed
                zarr_path = os.path.join(output_path,default_path,folder_name, 'agent_data_initial.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                agent_df = zarr_group_to_df(zarr_array,time_step=0)
                working_df=pd.merge(working_df, agent_df, on="AgentID")
                default_df = pd.concat([default_df, working_df], ignore_index=True)

null_df=pd.DataFrame()
for _,dirnames,_ in os.walk(os.path.join(output_path,null_path)):
    for folder_name in dirnames:
        if folder_name.startswith("null_"):
            print(f"Processing{os.path.join(output_path,null_path,folder_name,'agent_data.zarr')}")
            seed=int(folder_name.split('_')[-1])
            if seed in completed_seeds:
                zarr_path = os.path.join(output_path,null_path,folder_name,'agent_data.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                working_df = zarr_group_to_df(zarr_array, time_step=ts_target)
                working_df['seed']=seed
                zarr_path = os.path.join(output_path,null_path,folder_name, 'agent_data_initial.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                agent_df = zarr_group_to_df(zarr_array,time_step=0)
                working_df=pd.merge(working_df, agent_df, on="AgentID")
                null_df = pd.concat([null_df, working_df], ignore_index=True)

no_social_df=pd.DataFrame()
for _,dirnames,_ in os.walk(os.path.join(output_path,no_social_path)):
    for folder_name in dirnames:
        if folder_name.startswith("no_social_"):
            print(f"Processing{os.path.join(output_path,no_social_path,folder_name,'agent_data.zarr')}")
            seed=int(folder_name.split('_')[-1])
            if seed in completed_seeds:
                zarr_path = os.path.join(output_path,no_social_path,folder_name,'agent_data.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                working_df = zarr_group_to_df(zarr_array, time_step=ts_target)
                working_df['seed']=seed
                zarr_path = os.path.join(output_path,no_social_path,folder_name, 'agent_data_initial.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                agent_df = zarr_group_to_df(zarr_array,time_step=0)
                working_df=pd.merge(working_df, agent_df, on="AgentID")
                no_social_df = pd.concat([no_social_df, working_df], ignore_index=True)

no_adapt_df=pd.DataFrame()
for _,dirnames,_ in os.walk(os.path.join(output_path,no_adaptation_path)):
    for folder_name in dirnames:
        if folder_name.startswith("no_adapt_"):
            print(f"Processing{os.path.join(output_path,no_adaptation_path,folder_name,'agent_data.zarr')}")
            seed=int(folder_name.split('_')[-1])
            if seed in completed_seeds:
                zarr_path = os.path.join(output_path,no_adaptation_path,folder_name,'agent_data.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                working_df = zarr_group_to_df(zarr_array, time_step=ts_target)
                working_df['seed']=seed
                zarr_path = os.path.join(output_path,no_adaptation_path,folder_name, 'agent_data_initial.zarr')
                zarr_array = zarr.open(zarr_path, mode='r')
                agent_df = zarr_group_to_df(zarr_array,time_step=0)
                working_df=pd.merge(working_df, agent_df, on="AgentID")
                no_adapt_df = pd.concat([no_adapt_df, working_df], ignore_index=True)
'''
def calculate_histogram_data(df, column, bins):
    counts, bin_edges = np.histogram(df[column], bins=bins, density=True)
    percent_counts = counts * 100 
    return bin_edges[:-1], percent_counts

def save_histogram_data(dfs, labels, filename, column="wealth"):
    all_data = []
    for df, label in zip(dfs, labels):
        bins = np.arange(0, df[column].max() + 1, 1)
        bin_edges, percent_counts = calculate_histogram_data(df, column, bins)
        for bin_edge, percent_count in zip(bin_edges, percent_counts):
            all_data.append({"label": label, "bin": bin_edge, "frequency": percent_count})
    hist_data = pd.DataFrame(all_data)
    hist_data.to_csv(filename, index=False)

# Save histogram data for all scenarios in a single CSV file
dfs = [default_df, no_social_df, no_adapt_df, null_df]
labels = ["default_arrangement", "no_social_arrangement", "no_adapt_arrangement", "null_arrangement"]
save_histogram_data(dfs, labels, "wealth_histogram_data.csv")
'''

'''
def calculate_mutual_info(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    mi = mutual_info_regression(X, y)
    return mi

def process_mutual_info(df, label, target_col, feature_cols):
    mi = calculate_mutual_info(df, target_col, feature_cols)
    return label, mi

# Define the data and parameters
data_frames = {
    'Default': default_df,
    'No Social': no_social_df,
    'No Adaptation': no_adapt_df,
    'Null': null_df
}
target_col = 'wealth'
feature_cols = ['weighted_degree', 'i_a_accumulated', 'alpha', 'sigma', 'lambda', 'sensitivity', 'degree_initial', 'theta_initial', 'wealth_initial']

# Use parallel processing to calculate mutual information
results = Parallel(n_jobs=-1)(delayed(process_mutual_info)(df, label, target_col, feature_cols) for label, df in data_frames.items())

# Combine results into a DataFrame
mutual_info = pd.DataFrame({label: mi for label, mi in results}, index=feature_cols)

# Save to CSV
mutual_info.to_csv("mutual_info.csv")

'''


#Warning: the following section overwrites source dataframes with aggregated statistics

def percentile_aggregate(df, seed_col='seed', wealth_col='wealth', consumption_col='wealth_consumption'):
    # Calculate wealth percentiles
    df['wealth_percentile'] = df.groupby(seed_col)[wealth_col].transform(lambda x: pd.qcut(x, 10, labels=False))
    
    # Group by seed and wealth percentile, then calculate statistics
    grouped = df.groupby([seed_col, 'wealth_percentile']).agg(
        mean_wealth=(wealth_col, 'mean'),
        mean_consumption=(consumption_col, 'mean'),
        sd_wealth=(wealth_col, 'std'),
        sd_consumption=(consumption_col, 'std')
    ).reset_index()
    
    return grouped

default_df = percentile_aggregate(default_df)
no_social_df = percentile_aggregate(no_social_df)
no_adapt_df = percentile_aggregate(no_adapt_df)
null_df = percentile_aggregate(null_df)

# Add a column to identify the model arrangement
default_df['model'] = 'default_arrangement'
no_social_df['model'] = 'no_social_arrangement'
no_adapt_df['model'] = 'no_adapt_arrangement'
null_df['model'] = 'null_arrangement'

# Combine all results into a single DataFrame
combined_results = pd.concat([default_df, no_social_df, no_adapt_df, null_df], ignore_index=True)

# Save to CSV
combined_results.to_csv("percentile_wealth_consumption.csv", index=False)