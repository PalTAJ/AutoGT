

from base_operations.generalization import catGeneralization
from base_operations.auto_suppression import suppress_k_anonymity_violations,generqalizeBy_rules,pattern_based_generalization_fixed
from util import list_directories, clear_files, del_dir


import os
import pandas as pd
# from multiprocessing import Pool
from multiprocessing import Pool, freeze_support


def find_optimal_generalization_split_v3( df, column, target_column):
    rules = {}
    multi = [i for i in range(5, 105, 5)]
    catGen = catGeneralization()
    for n in multi:
        rules[n] = catGen.categorical_g_rule(df, column, target_column, n)

    return rules


def Extract_Rules(rules,rates):

    frules = {}
    
    for k,v in rates.items():
        if k in rules:
            frules[k] = rules[k][int(v)]

    return frules



def cluster_rows_v2(df, k, qis):
    """
    Finds exact k-anonymous groups based on QIs and returns:
    - List of row index lists for safe clusters
    - List of remaining row indices (to be generalized/suppressed)
    """
    clustered = []
    non_clustered = []

    grouped = df.groupby(qis).groups

    for _, indices in grouped.items():
        indices = list(indices)
        if len(indices) >= k:
            clustered.append(indices)
        else:
            non_clustered += indices

    return clustered, non_clustered


def is_fully_suppressed(series: pd.Series):
    """Checks if all values in a series are fully suppressed (i.e., only asterisks)."""
    unique_values = series.unique()
    return all(set(val) == {'*****'} for val in unique_values)


def has_any_suppressed(series: pd.Series, suppression_token='*****'):
    return series.apply(lambda x: set(str(x)) == suppression_token).any()

def count_suppressed(series: pd.Series, suppression_token='*****'):
    return series.apply(lambda x: set(str(x)) == suppression_token).sum()

def process_partition(args):
    df, qis, rules, avoid, k,numeric_supress = args
    # new_df = generqalizeBy_rules(df, qis, rule=rules, k=k, av=avoid)
    # new_df2 = suppress_k_anonymity_violations(new_df, qis, k)
    # return new_df2
    # print('--------------------------------------------------------\n')
    if len(numeric_supress) ==0:

        result_dfs = []

        # Step 1: Identify exact-match k-anonymous groups
        clusters, leftover_indices = cluster_rows_v2(df, k, qis)

        # Step 2: Add safe clusters with minimal generalization
        for cluster_indices in clusters:
            cluster_df = df.loc[cluster_indices]
            gen_cluster = generqalizeBy_rules(cluster_df, qis, rule=rules, k=k, av=avoid)
            result_dfs.append(gen_cluster)

        # Step 3: Handle remaining rows
        if leftover_indices:
            leftover_df = df.loc[leftover_indices]
            gen_leftover = generqalizeBy_rules(leftover_df, qis, rule=rules, k=k, av=avoid)
            suppressed_leftover = suppress_k_anonymity_violations(gen_leftover, qis, k)
            result_dfs.append(suppressed_leftover)
        final_df = pd.concat(result_dfs, ignore_index=True)
        return final_df

    else:

        result_dfs = []
        for col in numeric_supress:
            # df['long_str'] = df[col].apply(lambda x: str(x).replace('.', '').replace('-', ''))
            df[col] = pattern_based_generalization_fixed(df[col], k=k)
            # df = df.drop('long_str',axis=1)
            # print(df)


        # Step 1: Identify exact-match k-anonymous groups
        clusters, leftover_indices = cluster_rows_v2(df, k, qis)

        # Step 2: Add safe clusters with minimal generalization
        for cluster_indices in clusters:
            cluster_df = df.loc[cluster_indices]
            qis_safe = [q for q in qis if q not in numeric_supress]

            gen_cluster = generqalizeBy_rules(cluster_df, qis_safe, rule=rules, k=k, av=avoid,qis_num=numeric_supress)
            result_dfs.append(gen_cluster)

        # print(result_dfs)
        # Step 3: Handle remaining rows
        # print('----------------------------------------')
        if leftover_indices:
            leftover_df = df.loc[leftover_indices]
            # print(leftover_df)
            gen_leftover = generqalizeBy_rules(leftover_df, qis, rule=rules, k=k, av=avoid,qis_num=numeric_supress)
            qis_safe = [q for q in qis if q not in numeric_supress]
            suppressed_leftover = suppress_k_anonymity_violations(gen_leftover, qis_safe, k)
            result_dfs.append(suppressed_leftover)
            # print(suppressed_leftover)

        final_df = pd.concat(result_dfs, ignore_index=True)


        # print(final_df)
        return final_df

def  step_2_v22(partitions_dfs,qis, rules, avoid, k, target_column,file,numeric_supress):
    freeze_support()  # Required for Windows

    args_list = [(df, qis, rules, avoid, k,numeric_supress) for df in partitions_dfs]

    with Pool() as pool:
        result_dfs = pool.map(process_partition, args_list)

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df.to_csv(f'datasets/results/v3/{file}_results_latest_optimized_2025_{k}.csv', index=False)




################################################
def generate_approx_kanonymity_best_rates(partitions_dfs,file, sensitive_column,rates,qis,k=3,sg=[],qis_num=[],numeric_supress=[]):

    # numeric_supress = ['longitude', 'latitude']
    numeric_supress = []

    if len(numeric_supress) == 0 or numeric_supress[0] not in qis:
        numeric_supress = []


    cols = pd.read_csv(f"datasets/{file}.csv", nrows=1)
    cols = list(cols.columns)
    df = pd.read_csv(f"datasets/{file}.csv")
    # sg = []
    avoid = list(set(cols) - set(qis) - set(sg))


############################### Generate rules
    rules = {}
    for col in qis:
        if col not in sg :
            rule = find_optimal_generalization_split_v3(df, col, sensitive_column)
            rules[col] = rule

    #### numerical rules = []
    num_rules = {}
    groups_num = {}
    # for col in qis_num:
    #     groups = generate_number_groups(df, col, num_groups=groups_num[col])
    #     num_rules[col] = groups



    frules = Extract_Rules(rules,rates)
    output = step_2_v22(partitions_dfs,qis, frules, avoid, k, sensitive_column, file,numeric_supress)



def combine_files(file, k):
    print('------------ Combining Files...............')

    path_v3 = 'datasets/results/v3'

    filenames = list_directories(path_v3)
    filenames = [file for file in filenames if '.ipynb_checkpoints' not in file]

    ff = f'{file}_k_{k}_adapter'

    combined_csv2 = pd.concat([pd.read_csv(f'{path_v3}/{f}') for f in filenames])
    combined_csv2.to_csv(f"datasets/adapter_generated/{file}/{ff}.csv", index=False)
    print(f'\n \t path: datasets/adapter_generated/{ff}.csv \n')
    return f'datasets/adapter_generated/{ff}.csv'




