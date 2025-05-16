import pandas as pd


def get_columns(df):
    categorical = []
    names = list(df.columns)
    for name in names:
        if df.dtypes[name] == 'object':
            categorical.append(name)

    return [names,categorical]


def get_spans(df, partition, cats , scale=None):

    spans = {}
    for column in df.columns:
        if column in cats:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max() - df[column][partition].min()

        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def split(df, partition, cats, column):

    dfp = df[column][partition]
    if column in cats:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


def is_k_anonymous(df, partition, sensitive_column, k):

    if len(partition) < k:
        return False
    return True

def partition_dataset(df,feature_columns, sensitive_column, scale, is_valid,k=3):

    finished_partitions = []
    partitions = [df.index] # [RangeIndex(start=0, stop=48842, step=1)]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):    #median  of a certain column is used as splitting criteria
            lp, rp = split(df, partition, feature_columns,column)
            if not is_valid(df, lp, sensitive_column,k) or not is_valid(df, rp, sensitive_column,k):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def diversity(df, partition, column):
    return len(df[column][partition].unique())

def is_l_diverse(df, partition, sensitive_column, l=2):

    return diversity(df, partition, sensitive_column) >= l




def target_column_dist(df, sensitive_column):

    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p

    return global_freqs




def t_closeness(df, partition, column, global_freqs,k=3):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


def is_t_close(df,partition, sensitive_column, global_freqs, cats, p=0.2,k=3):

    if not sensitive_column in cats:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= p


