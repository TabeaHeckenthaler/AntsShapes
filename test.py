# from DataFrame.dataFrame import myDataFrame
# import pandas as pd
#
#
# def solving_percentage(size):
#     df = myDataFrame[(myDataFrame['size'] == size) &
#                                  (myDataFrame['shape'] == 'SPT') &
#                                  (myDataFrame['initial condition'] == 'back')]
#     print(pd.value_counts(df['winner']))
#
#
# if __name__ == '__main__':
#     solving_percentage('M')
#     DEBUG = 1

from Analysis.PathPy.Path import *

filename = 'large_20210419100024_20210419100547'
x = get(filename)
cs_labeled = ConfigSpace_AdditionalStates(x.solver, x.size, x.shape, x.geometry())
cs_labeled.load_labeled_space()
path = Path(time_step, x=x, conf_space_labeled=cs_labeled)
x.play(frames=[0, -50], path=path, videowriter=True)
