import pandas as pd

from collections import Counter
from util import list_directories
from base_operations.auto_suppression import generqalizeBy_rules,suppress_k_anonymity_violations, selective_generalize_to_k,pattern_based_generalization_fixed
import multiprocessing
from base_operations.informationLoss import target_col_corr

import os

import pandas as pd
import os
from multiprocessing import Pool

import pandas as pd



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

import pandas as pd
from multiprocessing import Pool, freeze_support




def process_partition(args):
    df, qis, rules, avoid, k,numeric_supress = args

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

        # print(result_dfs)

        final_df = pd.concat(result_dfs, ignore_index=True)
        # print(final_df)
        return final_df





def step_2_v3(partitions_dfs, qis, rules, avoid, k, target_column, rate, file,numeric_supress):
    freeze_support()  # Required for Windows

    args_list = [(df, qis, rules, avoid, k,numeric_supress) for df in partitions_dfs]

    with Pool() as pool:
        result_dfs = pool.map(process_partition, args_list)

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df.to_csv(f"datasets/temp_v3/stage2/k/{rate}.csv", index=False)









