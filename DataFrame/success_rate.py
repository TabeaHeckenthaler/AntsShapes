from DataFrame.Altered_DataFrame import Altered_DataFrame
import pandas as pd

shape = 'SPT'
solvers = ['ant']
geometry = ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')


for solver in solvers:
    geometry = None  # TODO
    columns = ['filename', 'winner', 'size', 'communication', 'average Carrier Number', 'time [s]']

    df = Altered_DataFrame()
    df.choose_experiments(solver=solver, shape=shape, geometry=geometry, init_cond='back')
    df.df = df.df[columns]

    df.df.to_json('success_rate.txt', orient='split')
    df = pd.read_json('success_rate.txt', orient='split')
