#This code generates data for two figures with four subplots of a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_regression,VarianceThreshold
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance


#The root path:
#output_path= "/Volumes/PTM_data/PTMOutput"
output_path= "output"
#The data folder paths:
arrangement_paths = ["SchemeA/default", "SchemeA/no_social","SchemeB/default", "SchemeB/no_social", "SchemeSQ/default", "SchemeSQ/no_social"]


completed_seeds=[15796,861,76821,54887,6266,
                82387,37195,87499,44132,60264,
                16024,41091,67222,64821,770,
                59736,62956,64926,67970,5312,
                83105,53708,85306,28694,71933,
                93017,25659,84479,18432,2748,
                59151,65726,84655,35774,67436,
                56887,66804,31552,11395,69093,
                3891,41607,96277,80039,87314,
                10628,8793,73970,43002,76553]           

#The target data:
#ts_target=[25,50,75]
#target_columns=False
ts_target=[50]
target_columns=["wealth"]
full_initial_data=False
accumulations=False





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
                print(f'sum:{np.sum(np.sum(zarr_array, axis=1))}')
            if array_name in ["theta","degree","wealth"] and full_initial_data!=False:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

# Create a dataframe from all available seeds for a model arrangement
# a target timestep specified above is used as a filter.

data_frames = {
    'SchemeA_default': pd.DataFrame(),
    'SchemeA_no_social': pd.DataFrame(),
    'SchemeB_default': pd.DataFrame(),
    'SchemeB_no_social': pd.DataFrame(),
    'SchemeSQ_default': pd.DataFrame(),
    'SchemeSQ_no_social': pd.DataFrame()

}
for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:
            if folder_name.startswith(f"{arrangement_path.split('/')[-1]}_"):
                seed=int(folder_name.split('_')[-1])
                if seed in completed_seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in ts_target:
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns=target_columns)
                        working_df['seed']=seed
                        working_df['Arrangement']=arrangement_path.split('/')[-1]
                        working_df['time_step']=ts
                        if full_initial_data==True:
                            zarr_path = os.path.join(output_path,arrangement_path,folder_name, 'agent_data_initial.zarr')
                            zarr_array = zarr.open(zarr_path, mode='r')
                            agent_df = zarr_group_to_df(zarr_array,time_step=0)
                            working_df=pd.merge(working_df, agent_df, on="AgentID")
                        data_frames[arrangement_path.replace('/', '_')] = pd.concat([data_frames[arrangement_path.replace('/', '_')], working_df], ignore_index=True)

dfs = list(data_frames.values())
labels = list(data_frames.keys())

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
            all_data.append({"Scheme":label.split("_")[0],"Arrangement": "_".join(label.split("_")[1:]), "bin": bin_edge, "frequency": percent_count})
    hist_data = pd.DataFrame(all_data)
    hist_data.to_csv(filename, index=False)

#Save histogram data for all scenarios in a single CSV file
save_histogram_data(dfs, labels, f"scheme_wealth_histogram_data_t{str(ts_target)}.csv")

summarydf = pd.DataFrame()
quantiledf = pd.DataFrame()
centiles = list(range(101))
for df, label in zip(dfs, labels):
    summary = pd.DataFrame({
        "Scheme": [label.split("_")[0]], 
        "Arrangement": ["_".join(label.split("_")[1:])],
        "Mean": [df["wealth"].mean()], 
        "Median": [df["wealth"].median()]
    })
    quantiles = pd.DataFrame({'Percentile':centiles,
                              "Value":[df["wealth"].quantile(c / 100.0) for c in centiles],
                              "Scheme":[label.split("_")[0]],
                              "Arrangement":["_".join(label.split("_")[1:])]})
    
    
    summarydf = pd.concat([summarydf, summary], ignore_index=True)
    quantiledf = pd.concat([quantiledf, quantiles], ignore_index=True)

summarydf.to_csv(f"scheme_wealth_histogram_data_t50_summary.csv")
quantiledf.to_csv(f"scheme_wealth_quantile_data_t50.csv")


'''
def compute_wasserstein_distance(dfs, labels, column="wealth"):
    distances = []
    for df, label in zip(dfs, labels):
        for seed in df['seed'].unique():
            seed_df = df[df['seed'] == seed]
            time_steps = sorted(seed_df['time_step'].unique())
            for t0, t1 in zip(time_steps[:-1], time_steps[1:]):
                bins = np.arange(0, seed_df[column].max() + 1, 1)
                bin_edges, counts_t0 = calculate_histogram_data(seed_df[seed_df['time_step'] == t0], column, bins)
                bin_edges, counts_t1 = calculate_histogram_data(seed_df[seed_df['time_step'] == t1], column, bins)
                sd=seed_df[seed_df['time_step'] == t1][column].std()
                distance = wasserstein_distance(counts_t0, counts_t1)
                distances.append({"Arrangement": label, "seed": seed, "t_0": t0, "t_1": t1, "Wasserstein_distance": distance, "k_t_1_sdev":sd})
    return pd.DataFrame(distances)

# Compute and save Wasserstein distances
wasserstein_distances = compute_wasserstein_distance(dfs, labels)
wasserstein_distances.to_csv(f"Wasserstein_distances.csv", index=False)



def calculate_mutual_info(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    #k-NN was problematic for zero variance features
    mi = np.zeros(len(feature_cols))
    filter = VarianceThreshold(threshold=0.0)
    X = filter.fit_transform(X)
    retained_indices = filter.get_support(indices=True)
    estimated_mi = mutual_info_regression(X, y)
    for i, index in enumerate(retained_indices):
        mi[index] = estimated_mi[i]
    return mi

def process_mutual_info(df, label, target_col, feature_cols, seed='All'):
    if seed != 'All':
        df = df[df['seed'] == seed]
    mi = calculate_mutual_info(df, target_col, feature_cols)
    return label, seed, mi

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
results = Parallel(n_jobs=-1)(delayed(process_mutual_info)(df, label, target_col, feature_cols, seed) for label, df in data_frames.items()
    for seed in completed_seeds + ['All'])

# Organize in df
mutual_info=[]
for label, seed, mi in results:
    for feature, value in zip(feature_cols, mi):
        mutual_info.append({'Arrangement': label, 'Seed': seed, 'Feature': feature, 'MI': value})
mutual_info = pd.DataFrame(mutual_info)

# Save to CSV
mutual_info.to_csv("mutual_info.csv")



'''

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
'''