from Analysis.Efficiency.PathLength import *
import os
from Directories import network_dir
from DataFrame.import_excel_dfs import df_ant_excluded
from itertools import groupby
import operator
from tqdm import tqdm

states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']


class PathLength_per_state(PathLength):
    def __init__(self, x):
        super().__init__(x)
        self.x.smooth(sec_smooth=2)
        self.ts = self.extend_time_series_to_match_frames(
            time_series_dict[self.x.filename], len(self.x.frames))

    def calculate_path_lengths_per_state(self, state, kernel_size=None):
        """
        Path length without counting any of the path length within the self.to_exclude states.
        """

        # find indices in which self.ts is state
        indices = [i for i, s in enumerate(self.ts) if s == state]

        # find the ranges of indices that are not equal to state
        ranges = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(operator.itemgetter(1), g))
            ranges.append((group[0], group[-1]))

        # cut out ranges from self.x
        translations = {}
        rotations = {}

        for r in ranges:
            x_new = self.x.cut_off(frame_indices=(r[0], r[1]))
            translations[str(r[0]) + '_' + str(r[1])] = PathLength(x_new).translational_distance(smooth=False)
            rotations[str(r[0]) + '_' + str(r[1])] = PathLength(x_new).rotational_distance(smooth=False)
        return translations, rotations

    @staticmethod
    def extend_time_series_to_match_frames(ts, frame_len):
        indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                             for _ in range(frame_len)]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\pathlengths_all_states.json'
    path_lengths = {}

    for index, row in tqdm(list(df_ant_excluded.iterrows())):
        traj = get(row['filename'])
        path_lengths[traj.filename] = {}
        print(traj.filename)
        pps = PathLength_per_state(traj)
        for state in tqdm(states, desc=traj.filename):
            path_lengths[traj.filename][state] = pps.calculate_path_lengths_per_state(state)
            # this is (translational, rotational)
        DEBUG = 1

    with open(direct, 'w') as json_file:
        json.dump(path_lengths, json_file)
        json_file.close()
