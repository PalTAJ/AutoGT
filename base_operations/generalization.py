from collections import Counter
import pandas as pd
from base_operations.informationLoss import theil_u,conditional_entropy
from collections import Counter, defaultdict
import numpy as np



################################################################

# auto generalization- catorigical and numerical V2

class catGeneralization:

    def __init__(self):
        pass

    def generate_ranges(self,df, y_column, x_column):

        rates = {}
        new_names = {}

        target_list = list(df[y_column].unique())
        for tv in range(1, len(target_list) + 1, 1):
            new_names[target_list[tv - 1]] = 'G' + str(tv)

        #     print(new_names)

        multi = [i for i in range(5, 105, 5)]
        for p in range(0, len(multi), 1):
            temp = {}

            for i in range(21):
                per = i * multi[p]
                if per <= 100:
                    temp[f'{x_column}{i}'] = (float(i) * multi[p])
            rates[multi[p]] = temp
        return [rates, new_names]

    def generate_r_v2(self,x, y, rates, n):

        data = {}
        fdata = {}
        #     n=15
        x_counter = Counter(x)  ## shows column data occurrences per uniqure value
        xy_counter = Counter(
            list(zip(x, y)))  ## we couple x and y values and count there occurrences (ex: [{(45,>=50):200},....])
        for xy in xy_counter.keys():  ## loop over xy coupled data
            p_xy = xy_counter[xy] / x_counter[xy[0]]  # find prob of coupled xy

            for k, i in rates[0][n].items():

                if (p_xy * 100.0 > i - 1 and p_xy * 100.0 < i + n - 1) or p_xy * 100.0 == 100.0:
                # if (p_xy*100.0 > i and p_xy*100.0 < i+n) or p_xy*100.0 == 100.0:


                    if xy[0] not in data:
                        data[xy[0]] = [i, p_xy * 100.0, xy[1], k + rates[1][xy[1]]]
                        fdata[xy[0]] = k + rates[1][xy[1]]

                    elif xy[0] in data and data[xy[0]][0] < i:
                        data[xy[0]] = [i, p_xy * 100.0, xy[1], k + rates[1][xy[1]]]
                        fdata[xy[0]] = k + rates[1][xy[1]]
                # else:
                    # print(xy,i)
        return fdata

    def categorical_g_rule(self,df, x_column, y_column, n):

        dist2 = {}

        df11 = df[x_column].tolist()
        df22 = df[y_column].tolist()

        ranges = self.generate_ranges(df, y_column, x_column)
        dist = self.generate_r_v2(df11, df22, ranges, n)
        # print(ranges)

        return dist


    def find_optimal_generalization_split(self,df, column, target_column):
        df2 = df.copy()

        df11 = df
        df22 = df2

        temp_u = 100.0   ## here changed
        temp_n = 0
        temp_df2 = 0
        temp_rules = {}
        rules = {}
        multi = [i for i in range(5, 105, 5)]

        for n in multi:

            rules = self.categorical_g_rule(df, column, target_column, n)

            df22 = df2.copy()
            # print(rules)
            df22[column] = df22[column].apply(lambda value: rules[value])
            u = theil_u(df11[column].tolist(), df22[column].tolist())
            # u = conditional_entropy(df11[column].tolist(), df22[column].tolist()) ## here changed
            # if u < temp_u:  ## here changed
                
            if u > temp_u:  ## here changed
                temp_n = n
                temp_u = u
                temp_df2 = df22
                temp_rules = rules

            # print(f'split percent: {n} entropy: {u}')
        df2 = temp_df2
        rules = temp_rules
        # print(column)
        # print(f'-------------\n best: \n split percent: {temp_n} entropy: {temp_u}\n')
        return rules

####################################################################







