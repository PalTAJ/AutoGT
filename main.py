
import pandas as pd
from util import list_directories, clear_files, del_dir
from base_operations.informationLoss import  target_col_corr
from base_operations.Kanonymity import get_columns, get_spans, target_column_dist, partition_dataset, is_k_anonymous, is_t_close, is_l_diverse
from base_operations.generalization import catGeneralization ,generate_number_groups
from automatic_operations.auto_generalization_v3 import step_2_v3

from k_v2 import generate_approx_kanonymity_best_rates
import os
import json
import copy

def find_optimal_generalization_split_v3( df, column, target_column):
    rules = {}
    multi = [i for i in range(5, 105, 5)]
    catGen = catGeneralization()
    for n in multi:
        rules[n] = catGen.categorical_g_rule(df, column, target_column, n)

    return rules


def dataset_settings(df ,name, columns):

    if 'housing' in name:
        sa = ['ocean_proximity']
        qis = ['latitude', 'longitude', 'housing_median_age', 'median_house_value', 'median_income']
        qi_index = get_qi_index(df ,qis)
        non_qis = [columns[i] for i in range(0, len(columns), 1) if i not in qi_index]


    elif 'adult' == name:
        qis = ['age', 'workclass', 'education', 'educational-num', 'marital-status', 'occupation', 'race',
               'gender', 'native-country']
        sa = ['income']
        qi_index = get_qi_index(df, qis)
        non_qis = [columns[i] for i in range(0, len(columns), 1) if i not in qi_index]

    elif 'mammographic_masses' in name:

        qis = ['BI-RADS assessment', 'Age', 'Shape', 'Margin', 'Density']
        sa = ['Severity']
        qi_index = get_qi_index(df, qis)
        non_qis = [columns[i] for i in range(0, len(columns), 1) if i not in qi_index]


    return [qis, non_qis, sa]


def get_qi_index(df ,columns):
    indexies = []
    for col in columns:
        ind = df.columns.get_loc(col)
        indexies.append(ind)
    return indexies


def generate_approx_kanonymity(file ,sensitive_column ,qis ,k=3 ,sg=[] ,qis_num=[]):



    # print(k)
    cols = pd.read_csv(f"datasets/{file}.csv", nrows=1)
    cols = list(cols.columns)
    df = pd.read_csv(f"datasets/{file}.csv")
    # print(df)

    ################3

    columns = list(df.columns)
    qis, non_qis, sa = dataset_settings(df ,file, columns)
    df2 = df[qis + non_qis]


    globalfreqs = target_column_dist(df2, sensitive_column)
    print('global freqs: ',globalfreqs)



    avoid = []

    partitions_dfs = []
    partitions_dfs_raw = []

    partitions_pth = f'partitions/housing/{k}/'
    partitions = os.listdir(partitions_pth)


    for partition in partitions:
        if '.csv' in partition:
            df2 = pd.read_csv(f'{partitions_pth}{partition}')

            df2 = df2.drop('ID',axis=1)
            partitions_dfs_raw.append(df2)



    print('done')
    ############################### Generate rules
    #### generates all rules based on all percentages

    rules = {}
    # sg.append(sensitive_column)
    for col in qis:
        # if col not in sg :
        rule = find_optimal_generalization_split_v3(df ,col ,sensitive_column)
        rules[col] = rule


    #### numerical rules = []
    # groups_num = {'median_income':4,'median_house_value':8,'housing_median_age':7}
    groups_num = {}
    #
    num_rules = {}
    for col in qis_num:
        groups = generate_number_groups(df, col, num_groups=groups_num[col])
        num_rules[col] = groups

    # print(sg)
    # print(rules)
    ################################ enforce-anonymity

    multi = [i for i in range(5, 105, 5)]
    for rate in multi:

        os.mkdir(f'datasets/temp_v3/stage1/{rate}')
        frules = {}

        # print(len(sg))
        for col ,val in rules.items():
            frules[col] = val[rate]

        print(rate)

        sg = list(set(sg))

        partitions_dfs = [copy.deepcopy(df) for df in partitions_dfs_raw]
        # step_2_v22(partitions_dfs,qis,frules,avoid,k,sensitive_column,rate,file)  # auto g and  suppression  v2
        # break
        # if __name__ == "__main__":
        # numeric_supress = ['longitude','latitude']
        numeric_supress = []

        if  len(numeric_supress)==0 or numeric_supress[0] not in qis :
            numeric_supress = []
        # print(frules)
        step_2_v3(partitions_dfs, qis, frules, avoid, k, sensitive_column, rate, file,numeric_supress)


    return partitions_dfs_raw




def combine_files(ss):

    multi = [i for i in range(5, 105, 5)]

    for rate in multi:
        # print(rate)
        filenames = list_directories(f'datasets/temp_v3/stage1/{rate}')
        filenames = [file for file in filenames if   '.ipynb_checkpoints' not in file ]
        # print(filenames)

        combined_csv = pd.concat([pd.read_csv(f'datasets/temp_v3/stage1/{rate}/{f}') for f in filenames])
        combined_csv.to_csv(f"datasets/temp_v3/stage2/{ss}/{rate}.csv", index=False)

    # filenames = list_directories(f'datasets/results/v3/')
    # combined_csv = pd.concat([pd.read_csv(f'datasets/results/v3/{f}') for f in filenames])
    # combined_csv.to_csv(f"datasets/temp_v3/stage2/{ss}/{rate}.csv", index=False)


def calculate_loss(senstive_column ,ss ,file):

    results = {}
    path = f"datasets/temp_v3/stage2/{ss}/"
    # dfm = pd.read_csv(f"datasets/{file}.csv")

    # o_corr = target_col_corr(dfm, senstive_column)
    # print('og--->',o_corr,'\n')

    filenames = list_directories(path)
    filenames = [file for file in filenames if   '.ipynb_checkpoints' not in file ]

    for file in filenames:
        rate = file.split('.')[0]
        df = pd.read_csv(f'{path}{file}')
        # print(df.columns)
        corr = target_col_corr(df ,senstive_column )
        results[rate] = corr
        # print(file.split('.')[0],rate,'--->',corr)
    # print(results)
    return results




def Find_Best_Percent(past_results ,qis):

    initial_cv = past_results['5'] ## init compare values
    final_rate = {}
    final_corr = {}
    # qis =  ['age', 'workclass', 'education', 'educational-num', 'marital-status', 'occupation', 'race',
    #        'gender', 'native-country']
    print(past_results)
    for qi in qis:
        for key ,value in past_results.items():
            if qi not in final_corr.keys():
                final_corr[qi] = value[qi]
                final_rate[qi] = key

            # elif final_corr[qi] > value[qi]:
            elif final_corr[qi] < value[qi]:  ## thail u

                final_corr[qi] = value[qi]
                final_rate[qi] = key
    # print()
    # print(final_rate)
    # print(final_corr)
    return final_rate

def Find_Best_Overall_Percent(past_results, qis):
    """
    Select a single generalization rate that minimizes total information loss across all QIs,
    and return a dictionary mapping each QI to that single best rate.
    """
    total_entropy_per_rate = {}

    for rate, entropy_dict in past_results.items():
        # Sum entropy values only for QIs that are present in this rate
        total_entropy = sum(entropy_dict[qi] for qi in qis if qi in entropy_dict)
        total_entropy_per_rate[rate] = total_entropy

    # Choose the rate with the minimum total entropy
    best_rate = min(total_entropy_per_rate, key=total_entropy_per_rate.get)

    # Return same format as your previous function — one rate for all QIs
    return {qi: best_rate for qi in qis}


def clear_paths():

    files = ['datasets/temp' ,'datasets/temp_v3/stage2/k' ,'datasets/results/v3']
    dirs = ['datasets/temp_v3/stage1']
    clear_files(files[0]) ;clear_files(files[1])
    del_dir(dirs[0])
    clear_files('datasets/results/v3')
    import time
    time.sleep(10)

def main():
    clear_paths()
    df_settings = pd.DataFrame(columns=['k', 'best_rules'])
    df_settings_counter = 0

    # file = 'mammographic_masses'
    # file = 'housing'
    file = 'adult'


    for k in range(2 ,3 ,1):

        print('K: ',k)
        clear_paths()
        # print(df_settings_counter)
        # sensitive_column = 'ocean_proximity'
        # sensitive_column = 'Severity'
        sensitive_column = 'income'


        # file = 'mammographic_masses'
        file = 'adult'
        # file = 'housing'

        ss = "k"

        qis = ['age', 'workclass', 'education', 'educational-num', 'marital-status', 'occupation', 'race',
               'gender', 'native-country']


        # qis = ['BI-RADS assessment','Age', 'Shape','Margin','Density' ]  ## mine

        # qis = [ 'housing_median_age', 'median_house_value', 'median_income','latitude','longitude']

        # 'median_income','median_house_value'
        # qis = [ 'latitude','longitude','housing_median_age','median_income','median_house_value']
        # qis = [ 'latitude','longitude','housing_median_age']

        # numeric_supress = ['longitude','latitude']
        numeric_supress = []

        qis_sg = []
        qis_num = []


        import time
        start = time.time()
        partitions = generate_approx_kanonymity(file ,sensitive_column,qis ,k=k ,sg = qis_sg ,qis_num=qis_num)
        # combine_files(ss)
        end = time.time()
        print('Full time per K',(end-start)/60)
        partitions_dfs = [copy.deepcopy(df) for df in partitions]

        results = calculate_loss(sensitive_column ,ss ,file)
        # print(results)

        # rates =Find_Best_Overall_Percent(results, qis)
        rates = Find_Best_Percent(results ,qis)   ## select best qi entropy per feature

        df_settings.loc[df_settings_counter] = [k ,str(rates)]
        df_settings_counter +=1
        df_settings.to_csv(f'{file}_3only_df_settings.csv' ,index=False)


        generate_approx_kanonymity_best_rates(partitions_dfs,file,sensitive_column ,rates ,qis ,k=k ,sg=qis_sg,numeric_supress=numeric_supress)
        #file, sensitive_column,rates,qis,k=3,sg=[],qis_num=[]
        path_v3 = 'datasets/results/v3'
        filenames = list_directories(path_v3)
        filenames = [file for file in filenames if   '.ipynb_checkpoints' not in file ]

        combined_csv2 = pd.concat([pd.read_csv(f'{path_v3}/{f}') for f in filenames])
        ff = f'final/{file}_test_best_optimized_{k}'

        combined_csv2.to_csv(f"datasets/{ff}.csv", index=False)
        #
        # # df_og = pd.read_csv(f"datasets/{file}.csv",sep=';')
        # df_og = pd.read_csv(f"datasets/{file}.csv")
        #
        # df = pd.read_csv(f"datasets/{ff}.csv")


    df_settings.to_csv(f'{file}_3only_df_settings.csv' ,index=False)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but recommended for Windows
    main()


