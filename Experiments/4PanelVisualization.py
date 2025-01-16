#This code generates plots with four subplots for a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
from sklearn.feature_selection import mutual_info_regression
from matplotlib.ticker import FixedLocator

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

graph_test_data=True


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

if graph_test_data==True:
    default_df=pd.read_csv("test_data_d.csv")
    null_df=pd.read_csv("test_data_n.csv")
    no_social_df=pd.read_csv("test_data_ns.csv")
    no_adapt_df=pd.read_csv("test_data_na.csv")
else:
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
    default_df.to_csv("test_data_d.csv")

    
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
    null_df.to_csv("test_data_n.csv")

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
    no_social_df.to_csv("test_data_ns.csv")

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
    no_adapt_df.to_csv("test_data_na.csv")

print(default_df.head())
print(no_social_df.head())
print(no_adapt_df.head())
print(null_df.head())

# Histogram of capital, k, at t=50
fig_hk, axs_hk = plt.subplots(2, 2, figsize=(10, 8))

axs_hk[0, 0].hist(default_df['wealth'], bins=np.arange(0, default_df["wealth"].max() + 1, 1), color='#342d49', density=True)
axs_hk[0, 0].set_title('(a) Default')
axs_hk[0, 0].set_xlim(0, 50)
axs_hk[0, 0].set_xlabel('Wealth, $k$')
axs_hk[0, 0].set_ylabel('Percent Frequency')
uncaptured = (default_df['wealth'] > 50).sum()/len(default_df['wealth'])*100
axs_hk[0, 0].text(0.95, 0.95, f'k > 50: {uncaptured:.2f}%', transform=axs_hk[0, 0].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

axs_hk[0, 1].hist(no_social_df['wealth'], bins=np.arange(0, no_social_df["wealth"].max() + 1, 1), color='#8b7d4f', density=True)
axs_hk[0, 1].set_title('(b) No Social')
axs_hk[0, 1].set_xlim(0, 50)
axs_hk[0, 1].set_xlabel('Wealth, $k$')
uncaptured = (no_social_df['wealth'] > 50).sum()/len(no_social_df['wealth'])*100
axs_hk[0, 1].text(0.95, 0.95, f'k > 50: {uncaptured:.2f}%', transform=axs_hk[0, 1].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

axs_hk[1, 0].hist(no_adapt_df['wealth'], bins=np.arange(0, no_adapt_df["wealth"].max() + 1, 1), color='#3f7bc1', density=True)
axs_hk[1, 0].set_title('(c) No Adaptation')
axs_hk[1, 0].set_xlim(0, 50)
axs_hk[1, 0].set_xlabel('Wealth, $k$')
axs_hk[1, 0].set_ylabel('Percent Frequency')
uncaptured = (no_adapt_df['wealth'] > 50).sum()/len(no_adapt_df['wealth'])*100
axs_hk[1, 0].text(0.95, 0.95, f'k > 50: {uncaptured:.2f}%', transform=axs_hk[1, 0].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

axs_hk[1, 1].hist(null_df['wealth'], bins=np.arange(0, null_df["wealth"].max() + 1, 1), color='#d15853', density=True)
axs_hk[1, 1].set_title('(d) Null')
axs_hk[1, 1].set_xlim(0, 50)
axs_hk[1, 1].set_xlabel('Wealth, $k$')
uncaptured = (null_df['wealth'] > 50).sum()/len(null_df['wealth'])*100
axs_hk[1, 1].text(0.95, 0.95, f'k > 50: {uncaptured:.2f}%', transform=axs_hk[1, 1].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

max_ylim = max(i.get_ylim()[1] for i in axs_hk.flat)
for i in axs_hk.flat:
    i.set_ylim(0, max_ylim)
    i.yaxis.set_major_locator(FixedLocator(i.get_yticks()))
    i.set_yticklabels([f'{100 * y:.0f}%' for y in i.get_yticks()])


plt.tight_layout()
plt.savefig("4Panel_hist_k.svg")

if graph_test_data==True:
    mutual_info=pd.read_csv("test_mutual_info.csv")
else:
    mutual_info = pd.DataFrame()
    #Discussion: It doesn't make sense to calculate MI for consumption, net trade, income, tech_index, or current i_a, degree, or theta does it?
    mutual_info['Default'] = calculate_mutual_info(default_df, 'wealth', ['weighted_degree','i_a_accumulated','alpha','sigma','lambda','sensitivity','degree_initial', 'theta_initial', 'wealth_initial'])
    mutual_info['No Social'] = calculate_mutual_info(no_social_df, 'wealth', ['weighted_degree','i_a_accumulated','alpha','sigma','lambda','sensitivity','degree_initial', 'theta_initial', 'wealth_initial'])
    mutual_info['No Adaptation'] = calculate_mutual_info(no_adapt_df, 'wealth', ['weighted_degree','i_a_accumulated','alpha','sigma','lambda','sensitivity','degree_initial', 'theta_initial', 'wealth_initial'])
    mutual_info['Null'] = calculate_mutual_info(null_df, 'wealth', ['weighted_degree','i_a_accumulated','alpha','sigma','lambda','sensitivity','degree_initial', 'theta_initial', 'wealth_initial'])
    mutual_info.index = ['weighted_degree','i_a_accumulated','alpha','sigma','lambda','sensitivity','degree_initial', 'theta_initial','wealth_initial']
    mutual_info.to_csv("test_mutual_info.csv")

fig_mi, axs_mi = plt.subplots(2, 2, figsize=(10, 8))

axs_mi[0, 0].barh(mutual_info.index, mutual_info['Default'], color='#342d49')
axs_mi[0, 0].set_title('(a) Default')
axs_mi[0, 0].set_xlabel('Mutual Information (nats)')
axs_mi[0, 0].set_xlim(0, 0.7)
axs_mi[0, 0].tick_params(axis='y')
axs_mi[0, 0].set_yticks(range(len(mutual_info.index)))
axs_mi[0, 0].set_yticklabels(['Weighted Degree', 'Gross Adaptation \nInvestment, $i_{a,total}$','Human Capital, $\\alpha$', 'Risk Aversion, $\\sigma$', 'Savings \nPropensity, $\\lambda$', 'Sensitivity', 'Initial Degree', 'Weighted Degree', 'Initial Wealth, $k_i$'],wrap=True)

axs_mi[0, 1].barh(mutual_info.index, mutual_info['No Social'], color='#8b7d4f')
axs_mi[0, 1].set_title('(b) No Social')
axs_mi[0, 1].set_xlabel('Mutual Information (nats)')
axs_mi[0, 1].set_xlim(0, 0.7)
axs_mi[0, 1].tick_params(axis='y')
axs_mi[0, 1].set_yticks(range(len(mutual_info.index)))
axs_mi[0, 1].set_yticklabels(['Weighted Degree', 'Gross Adaptation \nInvestment, $i_{a,total}$', 'Human Capital, $\\alpha$', 'Risk Aversion, $\\sigma$', 'Savings \nPropensity, $\\lambda$', 'Sensitivity', 'Initial Degree', 'Initial Shock \nPerception, $\\theta_i$', 'Initial Wealth, $k_i$'],wrap=True)

axs_mi[1, 0].barh(mutual_info.index, mutual_info['No Adaptation'], color='#3f7bc1')
axs_mi[1, 0].set_title('(c) No Adaptation')
axs_mi[1, 0].set_xlabel('Mutual Information (nats)')
axs_mi[1, 0].set_xlim(0, 0.7)
axs_mi[1, 0].tick_params(axis='y')
axs_mi[1, 0].set_yticks(range(len(mutual_info.index)))
axs_mi[1, 0].set_yticklabels(['Weighted Degree', 'Gross Adaptation \nInvestment, $i_{a,total}$', 'Human Capital, $\\alpha$', 'Risk Aversion, $\\sigma$', 'Savings \nPropensity, $\\lambda$', 'Sensitivity', 'Initial Degree', 'Initial Shock \nPerception, $\\theta_i$', 'Initial Wealth, $k_i$'],wrap=True)

axs_mi[1, 1].barh(mutual_info.index, mutual_info['Null'], color='#d15853')
axs_mi[1, 1].set_title('(d) Null')
axs_mi[1, 1].set_xlabel('Mutual Information (nats)')
axs_mi[1, 1].set_xlim(0, 0.7)
axs_mi[1, 1].tick_params(axis='y')
axs_mi[1, 1].set_yticks(range(len(mutual_info.index)))
axs_mi[1, 1].set_yticklabels(['Weighted Degree', 'Gross Adaptation \nInvestment, $i_{a,total}$', 'Human Capital, $\\alpha$', 'Risk Aversion, $\\sigma$', 'Savings \nPropensity, $\\lambda$', 'Sensitivity', 'Initial Degree', 'Initial Shock \nPerception, $\\theta_i$', 'Initial Wealth, $k_i$'],wrap=True)


plt.tight_layout()

plt.savefig("4Panel_mutual_info.svg")
