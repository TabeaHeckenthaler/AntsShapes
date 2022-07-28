from DataFrame.SingleExperiment import SingleExperiment
from Directories import SaverDirectories, df_minimal_dir
import os
import pandas as pd
from DataFrame.dataFrame import myDataFrame

filenames = [file for file in os.listdir(SaverDirectories['ps_simulation']) if file.startswith('minimal')]


class MinimalDataFrame(pd.DataFrame):
    def __init__(self, input):
        if type(input) is pd.DataFrame:
            super().__init__(input)

    @staticmethod
    def create_source():
        singleExperiments = []
        for filename in filenames:
            singleExperiments.append(SingleExperiment(filename, filename.split('_')[-1]))
        df = pd.concat(singleExperiments).reset_index(drop=True)
        df.to_json(df_minimal_dir)


    def create(self, df):




myMinimalDataFrame = MinimalDataFrame(pd.read_json(df_minimal_dir))

if __name__ == '__main__':
    MinimalDataFrame.create(myDataFrame)

myDataFrame['minimal path length [length unit]', ] = myDataFrame['filename'].map()