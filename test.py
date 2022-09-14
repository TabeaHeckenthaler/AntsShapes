from DataFrame.dataFrame import myDataFrame
import pandas as pd


def solving_percentage(size):
    df = myDataFrame[(myDataFrame['size'] == size) &
                                 (myDataFrame['shape'] == 'SPT') &
                                 (myDataFrame['initial condition'] == 'back')]
    print(pd.value_counts(df['winner']))


if __name__ == '__main__':
    solving_percentage('M')
    DEBUG = 1
