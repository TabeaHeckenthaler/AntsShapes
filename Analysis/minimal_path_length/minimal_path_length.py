from DataFrame.SingleExperiment import SingleExperiment
from Directories import SaverDirectories, df_minimal_dir, minimal_path_length_dir, home
import os
import pandas as pd
from DataFrame.dataFrame import myDataFrame
from Analysis.Efficiency.PathLength import PathLength
from trajectory_inheritance.get import get
import json
from DataFrame.import_excel_dfs import df_all, df_minimal
from os import path

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
        return minimal_path_length_dict

    def save_dict(self, minimal_path_length_dict):
        with open(minimal_path_length_dir, 'w') as json_file:
            json.dump(minimal_path_length_dict, json_file)
            json_file.close()

    @classmethod
    def get_dict(cls):
        with open(minimal_path_length_dir, 'r') as json_file:
            minimal_path_length_dict = json.load(json_file)
        return minimal_path_length_dict

    def update_dict(self, myDataFrame, m):
        to_add = set(myDataFrame['filename']) - set(m.keys())
        print(to_add)

        for series in myDataFrame[myDataFrame['filename'].isin(to_add)].iterrows():
            m.update({series[1]['filename']: self.find_minimal(series[1])})

        return m


# x = get('minimal_Large_SPT_back_MazeDimensions_human_LoadDimensions_human')
# PathLength(x).calculate_first_frame(sigma=0)
# minimal_path_length_dict = MinimalDataFrame.get_dict()
def find_minimal_filename(series):
    self = df_minimal
    if series['shape'] != 'SPT':
        return None
    if series['size'] == 'Small Near':
        series['size'] = 'Small Far'

    ind = self.loc[(self['size'] == series['size'])
                   & (self['initial condition'] == series['initial condition'])
                   & (self['maze dimensions'] == series['maze dimensions'])].index
    if len(ind) > 1:
        if series['solver'] == 'human' and self.loc[ind[0]]['size'].split(' ')[0] == 'Small':
            return self.loc[ind[0]]['filename']
        print(series['filename'])
        raise ValueError('too many minimal trajectories')
    if len(ind) == 1:
        result = self.loc[ind]['filename'].iloc[0]
        return result


if __name__ == '__main__':
    df_all['minimal filename'] = df_all.apply(find_minimal_filename, axis=1)
    minimal_filename_dict = dict(zip(df_all['filename'], df_all['minimal filename']))
    minimal_path_length_dir = path.join(home, 'Analysis', 'minimal_path_length', 'minimal_filename_dict.json')

    with open(minimal_path_length_dir, 'w') as json_file:
        json.dump(minimal_filename_dict, json_file)
        json_file.close()

    DEBUG = 1
    m = MinimalDataFrame()
    # m.to_excel(lists_exp_dir + '\\exp_minimal.xlsx')

    MinimalDataFrame.create_source()
    myMinimalDataFrame = MinimalDataFrame()
    # myMinimalDataFrame.create_dict()
    minimal_path_length_dict = MinimalDataFrame.get_dict()
    minimal_path_length_dict = myMinimalDataFrame.update_dict(myDataFrame, minimal_path_length_dict)
    myMinimalDataFrame.save_dict(minimal_path_length_dict)
    DEBUG = 1
