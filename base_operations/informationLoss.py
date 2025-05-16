

import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt




def conditional_entropy(x, y):
    # entropy of x given y
    ## x is a list of senstive column data (income in our case)
    ## y is list of other column data

    y_counter = Counter(y)  ## shows column data occurrences per uniqure value

    xy_counter = Counter(
        list(zip(x, y)))  ## we couple x and y values and count there occurrences (ex: [{(45,>=50):200},....])
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():  ## loop over xy coupled data
        p_xy = xy_counter[xy] / total_occurrences  # find prob of coupled xy
        p_y = y_counter[xy[1]] / total_occurrences  # find prob of y(single value, example here its age value)
        entropy += p_xy * math.log(p_y / p_xy)  # Entropy equation  = p(x,y)* log(p(y)/p(x,y))
    return entropy


def theil_u(x, y):
    #     calcualte theil_u correlation  using entropy

    s_xy = conditional_entropy(x, y)  # calculates entropy
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences,
                   x_counter.values()))  # p(x), x column values occurances divided by total occurance
    s_x = ss.entropy(p_x)  # calculate entropy

    #  Calculate the entropy of a distribution for given probability values.
    #  If only probabilities pk(fist argument of the fn) are given, the entropy is calculated as S = -sum(pk * log(pk), axis=axis).
    # otherwise if two are given it calculate normal entropy

    if s_x == 0:  ## since equation will return -sum(....) we return 1, S = -sum(pk * log(pk)
        return 1
    else:
        return (s_x - s_xy) / s_x  ## Uncertainty_coefficient equation(H(x) is entropy of x) = (H(x)-H(xy)) / H(x)



def target_col_corr(df, target):

    results = {}
    # theilu = pd.DataFrame(index=[target], columns=df.columns)
    columns = df.columns
    for j in range(0, len(columns)):
        # u = conditional_entropy(df[target].tolist(), df[columns[j]].tolist())
        u = theil_u(df[target].tolist(), df[columns[j]].tolist())

        results[columns[j]] = u
        # print(columns[j], '---->', u)

    return results


