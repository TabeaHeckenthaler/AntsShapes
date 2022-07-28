from DataFrame.SingleExperiment import SingleExperiment
from Directories import SaverDirectories, df_minimal_dir, minimal_path_length_dir
import os
import pandas as pd
from DataFrame.dataFrame import myDataFrame
from Analysis.PathLength.PathLength import PathLength
from trajectory_inheritance.get import get
import json

missing = set()


class MinimalDataFrame(pd.DataFrame):
    def __init__(self):
        super().__init__(pd.read_json(df_minimal_dir))

    @staticmethod
    def create_source():
        singleExperiments = []
        filenames = MinimalDataFrame.get_minimal_filenames()
        for filename in filenames:
            s = SingleExperiment(filename, filename.split('_')[-1])
            x = get(s['filename'][0])
            s['path length [length unit]'] = float(PathLength(x).calculate_path_length(sigma=0, penalize=False))
            singleExperiments.append(s)
        df = pd.concat(singleExperiments).reset_index(drop=True)
        df.to_json(df_minimal_dir)

    @classmethod
    def get_minimal_filenames(cls):
        return [file for file in os.listdir(SaverDirectories['ps_simulation']) if file.startswith('minimal')]

    def find_minimal(self, series):
        if series['shape'] != 'SPT':
            return None
        if series['size'] == 'Small Near':
            series['size'] = 'Small Far'

        ind = self.loc[(self['size'] == series['size'])
                       & (self['initial condition'] == series['initial condition'])
                       & (self['maze dimensions'] == series['maze dimensions'])].index
        if len(ind) > 1:
            if series['solver'] == 'human' and self.loc[ind[0]]['size'].split(' ')[0] == 'Small':
                return self.loc[ind[0]]['path length [length unit]']
            print(series['filename'])
            raise ValueError('too many minimal trajectories')
        if len(ind) == 1:
            result = self.loc[ind]['path length [length unit]'].iloc[0]
            return result

        print(series['size'] + series['initial condition'] + series['maze dimensions'] + ' not in minimal')
        # missing.add(series['size'] + series['initial condition'] + series['maze dimensions'])
        return None

    def create_dict(self):
        myDataFrame['minimal path length [length unit]'] = myDataFrame.apply(self.find_minimal, axis=1)
        minimal_path_length_dict = {f: N for f, N in zip(myDataFrame['filename'],
                                                         myDataFrame['minimal path length [length unit]'])}

        with open(minimal_path_length_dir, 'w') as json_file:
            json.dump(minimal_path_length_dict, json_file)
            json_file.close()

#
# x = get('minimal_Large_SPT_back_MazeDimensions_human_LoadDimensions_human')
# PathLength(x).calculate_path_length(sigma=0)


with open(minimal_path_length_dir, 'r') as json_file:
    minimal_path_length_dict = json.load(json_file)




if __name__ == '__main__':
    DEBUG = 1
    # MinimalDataFrame.create_source()
    # myMinimalDataFrame = MinimalDataFrame()
    # myMinimalDataFrame.create_dict()
    # DEBUG = 1
