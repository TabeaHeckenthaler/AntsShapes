from DataFrame.choose_experiments import Altered_DataFrame
import pandas as pd

shape = 'SPT'
solvers = ['ant']
geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')


for solver in solvers:
    geometry = None  # TODO
    columns = ['filename', 'winner', 'size', 'communication', 'average Carrier Number', 'time [s]']

    df = Altered_DataFrame()
    df.choose_experiments(shape, solver, geometry, init_cond='back')
    df.choose_columns(columns)

    df.df.to_json('success_rate.txt', orient='split')
    df = pd.read_json('success_rate.txt', orient='split')
