from DataFrame.dataFrame import choose_relevant_experiments, myDataFrame
import os
import json
import pandas as pd

shape = 'SPT'
solvers = ['ant']


def relevant_columns(df):
    columns = ['filename', 'winner', 'size', 'communication', 'average Carrier Number', 'time [s]']
    df = df[columns]
    return df


for solver in solvers:
    df_relevant_exp = choose_relevant_experiments(myDataFrame.clone(), shape, solver, geometry, init_cond='back')
    df = relevant_columns(df_relevant_exp)

    df.to_json('success_rate.txt', orient='split')
    df = pd.read_json('success_rate.txt', orient='split')
