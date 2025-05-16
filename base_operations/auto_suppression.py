
import pandas as pd


def suppress_k_anonymity_violations(df, qis, k, suppression_value="*****"):
    """
    Converts QI columns to object type if needed and suppresses values in violating rows.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - qis (list of str): List of quasi-identifier columns.
    - k (int): Minimum group size to satisfy k-anonymity.
    - suppression_value (str): Suppression string to insert.

    Returns:
    - pd.DataFrame: Modified DataFrame with suppressed QIs.
    """
    df_copy = df.copy()
    group_sizes = df_copy.groupby(qis).size()
    violating_groups = group_sizes[group_sizes < k].index.tolist()

    if violating_groups:
        mask = df_copy[qis].apply(tuple, axis=1).isin(violating_groups)

        # Only convert columns if we need to suppress
        for col in qis:
            if mask.any():
                df_copy[col] = df_copy[col].astype('object')
                df_copy.loc[mask, col] = suppression_value

    return df_copy

def selective_generalize_to_k(values, k=3, max_digits=5):
    """
    Progressively generalize string-encoded numeric values until all groups satisfy k-anonymity.
    """
    for level in range(1, max_digits + 1):
        values = values.astype(str)
        gen = values.apply(lambda v: v[:-level] + '*' * level if len(v) > level else '*' * len(v))
        value_counts = gen.value_counts()

        # Check if all groups satisfy k-anonymity
        if (value_counts >= k).all():
            return gen

    return gen  # most generalized version



def generqalizeBy_rules(df, qis,rule={},k=3,auto_rule={}, auto=0, av = [],sg = [],qis_num=[]):

    sg +=av
    sg+=qis_num

    if len(df) > 0:

        for col in qis:
            if col not in qis_num  and len(df[col].unique()) > 1:
                print(col)
                print(rule[col])
                df[col] = df[col].apply(lambda value: rule[col][value])
                # df[col] = df[col].astype(str).apply(lambda value: rule[col][value])

    return df



def longestCommonPrefix2(arr):
    if not arr:
        return ""

    result = arr[0]
    length = len(result)

    for i in range(1, len(arr)):
        while arr[i].find(result) != 0:
            result = result[:length - 1]
            length -= 1
            if not result:
                return ""
    return result


def pattern_based_generalization_fixed(values: pd.Series, k=3):
    str_vals = values.astype(str).tolist()

    # Find longest common prefix
    prefix = longestCommonPrefix2(str_vals)

    # Apply generalization
    gen_values = []
    for val in str_vals:
        masked = prefix
        gen_values.append(masked)

    # Count group sizes
    gen_series = pd.Series(gen_values)
    group_sizes = gen_series.value_counts()
    group_map = gen_series.map(group_sizes)

    # Check if all groups have ≥ k members
    if (group_map >= k).all():
        return gen_series
    else:
        return pd.Series(['*' * len(v) for v in str_vals])