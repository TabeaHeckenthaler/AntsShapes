from itertools import groupby
import numpy as np
from trajectory_inheritance.get import get
from trajectory_inheritance.exp_types import solver_geometry
import pandas as pd
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Labeled, pre_final_state
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
# from Analysis.Efficiency.PathLength import PathLength
from Setup.Maze import Maze
from PhysicsEngine.Display import Display
from tqdm import tqdm
from Directories import network_dir
import os
import json
from matplotlib import pyplot as plt
from DataFrame.plot_dataframe import save_fig
from DataFrame.import_excel_dfs import df_relevant, dfs_ant_old, dfs_ant, dfs_human, dfs_pheidole

time_step = 0.25  # seconds

color_dict = {'b': '#9700fc', 'be': '#e0c1f5', 'b1': '#d108ba', 'b2': '#38045c',
              # 'bf': '#d108ba',
              # 'a': '#fc0000',
              'ac': '#fc0000', 'ab': '#802424',
              'c': '#fc8600', 'cg': '#8a4b03',
              'e': '#fcf400', 'eb': '#a6a103', 'eg': '#05f521',
              'f': '#30a103', 'g': '#000000',
              'h': '#085cd1', False: '#000000', True: '#ccffcc', 'i': '#000000',}

c = {'edge-moving': '#00ff00',
     'free-moving': '#004d00',
     'edge-still': '#e60099',
     'free-still': '#4d0033'
     }

color_dict.update(c)

def exp_day(filename):
    if filename.startswith('sim'):
        return filename
    if len(filename.split('_')) == 3:
        return '_'.join(filename.split('_')[:3])
    return '_'.join(filename.split('_')[:3]) + '_' + filename.split('_')[4]


def create_bar_chart(df: pd.DataFrame, ax: plt.Axes, block=False, sorted=False):
    if sorted:
        df = df.sort_values('time [s]')

    # for filename, ts, winner, food in zip(df['filename'], df['time series'], df['winner'], df['food in back']):
    for filename, ts in zip(df['filename'], df['time series']):
        p = Path(time_step=0.25, time_series=ts)
        print(filename)
        if filename == 'M_SPT_4340004_MSpecialT_1_ants':
            DEBUG = 1
        p.bar_chart(ax=ax, axis_label=exp_day(filename), block=block)
        if not block:
            ax.set_xlabel('time [min]')
        else:
            ax.set_xlabel('')
        # ax.set_xlim([0, 20])
    DEBUG = 1


class Path:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """

    def __init__(self, time_step: float, time_series=None, x=None, conf_space_labeled=None, only_states=False):
        """
        param step: after how many frames I add a label to my label list
        :param x: trajectory
        :return: list of strings with labels
        """
        self.time_step = time_step
        self.time_series = time_series
        if self.time_series is None and x is not None:
            self.time_series = self.get_time_series(conf_space_labeled, x)
            # print('No correction')
            self.time_series = conf_space_labeled.correct_time_series(self.time_series, filename=x.filename)
            # self.save_transition_images(x)
            if only_states:
                self.time_series = [l[0] for l in self.time_series]
        self.state_series = self.calculate_state_series(self.time_series)
        DEBUG = 1
        # self.state_series = None

    @staticmethod
    def show_configuration(x, coords, save_dir='', text_in_snapshot='', frame=0):
        maze = Maze(x)
        display = Display(text_in_snapshot, 1, maze, frame=frame)
        maze.set_configuration(coords[0:2], coords[2])
        maze.draw(display)
        display.display()
        if len(save_dir) > 0:
            display.snapshot(save_dir)

    def save_transition_images(self, x):
        coords = [coords for coords in x.iterate_coords_for_ps()]
        Path.show_configuration(x, coords[0],
                                'State_transition_images\\' + str(int(0 * self.time_step)) + '.png',
                                text_in_snapshot=self.time_series[0], frame=0)
        for i, (label0, label1) in enumerate(zip(self.time_series[:-1], self.time_series[1:])):
            if label0 != label1:
                Path.show_configuration(x, coords[i],
                                        'State_transition_images\\' + str(int(i * self.time_step)) + '.png',
                                        text_in_snapshot=label1, frame=int(i * self.time_step * x.fps))

    def get_time_series(self, conf_space_labeled, x):
        if conf_space_labeled.adapt_to_new_dimensions:
            x = x.confine_to_new_dimensions()
            # x.play(geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
        coords_in_cs = [coords for coords in x.iterate_coords_for_ps(time_step=self.time_step)]
        # coo = (4.2649902310867205, 4.834794427139041, 4.863976661556442)
        # conf_space_labeled.coords_to_indices(*coo, shape=conf_space_labeled.space_labeled.shape)
        indices = [conf_space_labeled.coords_to_indices(*coords, shape=conf_space_labeled.space_labeled.shape)
                   for coords in coords_in_cs]
        labels = [None]
        # maze = Maze(x, geometry=conf_space_labeled.geometry)
        for i, index in tqdm(enumerate(indices)):
            # if i == 889:
            #     DEBUG = 1
            labels.append(self.label_configuration(index, conf_space_labeled))
            # print(labels[-1])
            # display = Display(labels[-1], 1, maze, frame=i)
            # maze.set_configuration(coords[i][0:2], coords[i][2])
            # maze.draw(display)
            # display.display()
        labels = labels[1:]

        if pre_final_state in labels[-1] and pre_final_state != labels[-1]:
            labels.append(pre_final_state)

        return labels

    def label_configuration(self, index, conf_space_labeled) -> str:
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            DEBUG = 1
            label = conf_space_labeled.find_closest_state(index)
        # if set(label) == set('d'):
        #     conf_space_labeled.draw_ind(index)
        return label

    @staticmethod
    def combine_transitions(labels) -> list:
        """
        I want to combine states, that are [.... 'gb' 'bg'...] to [... 'gb'...]
        """
        labels = [''.join(sorted(state)) for state in labels]
        mask = [True] + [sorted(state1) != sorted(state2) for state1, state2 in zip(labels, labels[1:])]
        return np.array(labels)[mask].tolist()

    @staticmethod
    def symmetrize(state_series):
        return [state.replace('d', 'e') for state in state_series]

    @staticmethod
    def only_states(state_series):
        return [state[0] for state in state_series]

    def state_at_time(self, time: float) -> str:
        if int(time / self.time_step) > len(self.time_series):
            return None
        return self.time_series[int(time / self.time_step)]

    # def interpolate_zeros(self, labels: list) -> list:
    #     """
    #     Interpolate over all the states, that are not inside Configuration space (due to the computer representation of
    #     the maze not being exactly the same as the real maze)
    #     :return:
    #     """
    #
    #     if labels[0] == '0':
    #         labels[0] = [l for l in labels if l != '0'][:1000][0]
    #
    #     for i, l in enumerate(labels):
    #         if l == '0':
    #
    #             where = np.where(np.array(labels[i:]) != '0')[0]
    #
    #             if len(where) < 1:
    #                 if labels[i - 1] == final_state + pre_final_state:
    #                     labels[i:] = [final_state for _ in range(len(labels[i:]))]
    #                 else:
    #                     labels[i:] = [labels[i - 1] for _ in range(len(labels[i:]))]
    #
    #             else:
    #                 index = where[0] + i
    #                 if self.valid_transition(labels[i - 1], labels[index]):
    #                     transitions = [labels[i - 1], labels[index]]
    #                 else:
    #                     if labels[i - 1] == 'e' and labels[index] == 'g':
    #                         transitions = ['e', 'e']  # this occurs only in small SPT ants
    #                     else:
    #                         transitions = [labels[i - 1], *self.necessary_transitions(labels[i - 1], labels[index]),
    #                                        labels[index]]
    #
    #                 for array, state in zip(np.array_split(range(i, index), len(transitions)), transitions):
    #                     if len(array > 0):
    #                         labels[array[0]: array[-1] + 1] = [state for _ in range(len(array))]
    #
    #     if len([1 for state1, state2 in zip(labels[:-1], labels[1:]) if
    #             state1 == final_state and state2 != final_state]):
    #         print('something')
    #
    #     return labels

    @staticmethod
    def calculate_state_series(time_series: list):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        if time_series is None:
            return None
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series])]
        # labels = Path.combine_transitions(labels)
        return labels

    @staticmethod
    def time_stamped_series(time_series, time_step) -> list:
        groups = groupby(time_series)
        return [(label, sum(1 for _ in group) * time_step) for label, group in groups]

    #
    # @staticmethod
    # def path_length_stamped_series(filename, time_series, time_step) -> list:
    #     x = get(filename)
    #     hey = Path.time_stamped_series(time_series, time_step)
    #     frames = [(h[0], int(h[1] * x.fps)) for h in hey]
    #     hey_path_length = []
    #     current = 0
    #
    #     for (state, frame_number) in frames:
    #         new_end_frame = min(current + frame_number, len(x.frames))
    #         path_l = PathLength(x).calculate_path_length(frames=[current, new_end_frame])
    #         current = new_end_frame
    #         hey_path_length.append((state, path_l))
    #     # x.play()
    #     return hey_path_length

    @classmethod
    def create_dicts(cls, df_all, ConfigSpace_class=ConfigSpace_SelectedStates, dictio_ts=None, dictio_ss=None):
        if dictio_ts is None:
            dictio_ts = {}
        if dictio_ss is None:
            dictio_ss = {}
        shape = 'SPT'
        df_all = df_all[df_all['shape'] == shape]

        for solver in df_all['solver'].unique():
            print(solver)
            df = df_all[df_all['solver'] == solver].sort_values('size')
            groups = df.groupby(by=['size', 'maze dimensions', 'load dimensions'])
            for (size, maze_dim, load_dim), cs_group in groups:
                cs_labeled = ConfigSpace_class(solver, size, shape, (maze_dim, load_dim))
                cs_labeled.load_final_labeled_space()
                for _, exp in tqdm(cs_group.iterrows()):
                    with open("state_calc.txt", "a") as f:
                        f.write('\n' + exp['filename'])
                    print(exp['filename'])
                    if exp['filename'] not in dictio_ts.keys():
                        # if (exp['maze dimensions'], exp['load dimensions']) != solver_geometry[solver]:
                        #     dictio_ts[exp['filename']] = None
                        #     dictio_ss[exp['filename']] = None
                        # else:
                            # if exp['filename'] == 'L_SPT_4080010_SpecialT_1_ants (part 1)':
                            #     DEBUG = 1
                        # x = get('L_SPT_4080002_SpecialT_1_ants (part 1)')
                        # print('L_SPT_4080002_SpecialT_1_ants (part 1)')
                        x = get(exp['filename'])
                        path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                        dictio_ts[exp['filename']] = path_x.time_series
                        dictio_ss[exp['filename']] = path_x.state_series
                        # dictio_ts[exp['filename']] = None
                        # dictio_ss[exp['filename']] = None
            return dictio_ts, dictio_ss

    @classmethod
    def plot_paths(cls, zipped, time_series_dict):
        for solver, df_dict in zipped:
            for size, df in df_dict.items():
                fig, ax = plt.subplots()
                df['time series'] = df['filename'].map(time_series_dict)

                create_bar_chart(df, ax, block=False)
                save_fig(fig, 'bar_chart_' + solver + '_' + str(size))

        # shape = 'SPT'
        # df_all = df_all[df_all['shape'] == shape]

        # for solver in df_all['solver'].unique():
        #     print(solver)
        #     df = df_all[df_all['solver'] == solver].sort_values('size')
        #     groups = df.groupby(by=['size'])
        #     for size, cs_group in groups:
        #         fig, ax = plt.subplots()
        #         plot_bar_chart(cs_group, ax, block=False)
        #         save_fig(fig, 'bar_chart_' + solver + '_' + str(size))

    @staticmethod
    def find_missing(myDataFrame, solver=None):

        DEBUG = 1

        myDataFrame = myDataFrame[myDataFrame['shape'] == 'SPT']
        myDataFrame = myDataFrame[myDataFrame['solver'] == 'human']
        myDataFrame = myDataFrame[myDataFrame['size'].isin(['Small Far', 'Small Near'])]
        myDataFrame = myDataFrame[~myDataFrame['filename'].str.contains('free')]
        if solver is not None:
            myDataFrame = myDataFrame[myDataFrame['solver'] == solver]

        to_add = myDataFrame[~myDataFrame['filename'].isin(time_series_dict.keys())]
        return to_add

    @classmethod
    def add_to_dict(cls, to_add, ConfigSpace_class, time_series_dict, state_series_dict, solver=None) -> tuple:
        """

        """
        dictio_ts = {}
        dictio_ss = {}

        solver_groups = to_add.groupby('solver')
        for solver, solver_group in solver_groups:
            size_groups = solver_group.groupby('size')
            for size, cs_group in size_groups:
                print(size)
                cs_labeled = ConfigSpace_class(solver, size, 'SPT', solver_geometry[solver])
                cs_labeled.load_final_labeled_space()
                for _, exp in tqdm(cs_group.iterrows()):
                    print(exp['filename'])
                    if (exp['maze dimensions'], exp['load dimensions']) != solver_geometry[solver]:
                        dictio_ts[exp['filename']] = None
                        dictio_ss[exp['filename']] = None
                    else:
                        x = get(exp['filename'])
                        path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                        dictio_ts[exp['filename']] = path_x.time_series
                        dictio_ss[exp['filename']] = path_x.state_series
        time_series_dict.update(dictio_ts)
        state_series_dict.update(dictio_ss)
        return time_series_dict, state_series_dict

    def bar_chart(self, ax, axis_label='', block=False, array=None):
        if array is None:
            ts = self.time_series
            if 'i' in ts:
                ts.remove('i')
        else:
            ts = array
        # ts = Path.only_states(self.time_series)
        # ts = Path.symmetrize(ts)
        # dur = Path.state_duration(ts)
        dur = Path.time_stamped_series(ts, self.time_step)

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # color_dict = {state: color for state, color in zip(['a', 'b', 'c', 'd', 'h', 'f'], colors)}

        left = 0
        given_names = {}

        # change plt fontsize
        plt.rcParams.update({'font.size': 12})

        for name, duration in dur:
            dur_in_min = duration / 60
            if block:
                b = ax.barh(axis_label, 1, color=color_dict[name], left=left, label=name)
                left += 1
            else:
                b = ax.barh(axis_label, dur_in_min, color=color_dict[name], left=left, label=name)
                left += dur_in_min
            if name not in given_names:
                given_names.update({name: b})

        # if winner:
        #     plt.text(left + 1, b.patches[0].xy[-1], 'v', color='green')
        # else:
        #     plt.text(left + 1, b.patches[0].xy[-1], 'x', color='red')
        #
        # if food:
        #     plt.text(left + 2, b.patches[0].xy[-1], 'f', color='black')

        # labels = list(color_dict.keys())
        labels = list(given_names.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in labels]
        plt.legend(handles, labels)

    @staticmethod
    def get_dicts(name='') -> tuple:
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'r') as json_file:
            time_series_dict = json.load(json_file)
            json_file.close()

        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'r') as json_file:
            state_series_dict = json.load(json_file)
            json_file.close()
        return time_series_dict, state_series_dict

    @staticmethod
    def save_dicts(time_series_dict, state_series_dict, name=''):
        print('saving in', os.path.join(network_dir, 'time_series' + name + '.json'))
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'w') as json_file:
            json.dump(time_series_dict, json_file)
            json_file.close()

        print('saving in', os.path.join(network_dir, 'state_series' + name + '.json'))
        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'w') as json_file:
            json.dump(state_series_dict, json_file)
            json_file.close()
        print('saved!')


# time_series_dict, state_series_dict = Path.get_dicts('_selected_states')

# empty = [k for k, v in time_series_dict.items() if v is None]
# for e in empty:
#     del time_series_dict[e]
#     del state_series_dict[e]

DEBUG = 1
if __name__ == '__main__':
    filename = 'S_SPT_4350024_SSpecialT_1_ants'
    x = get(filename)
    x.geometry()
    x_new = x.confine_to_new_dimensions()
    x_new.play(geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
    cs_labeled = ConfigSpace_SelectedStates('ant', x.size, 'SPT', x.geometry())
    cs_labeled.load_final_labeled_space()
    path = Path(time_step, x=x, conf_space_labeled=cs_labeled)

    # with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'w') as json_file:
    #     json.dump(time_series_dict, json_file)
    #     json_file.close()
    #
    # with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
    #     state_series_dict = json.load(json_file)
    #     json_file.close()
    #

    # filename = 'small2_20201221102928_20201221103004_20201221103004_20201221103244'
    # x = get(filename)
    # cs_labeled = ConfigSpace_AdditionalStates(x.solver, x.size, x.shape, x.geometry())
    # cs_labeled.load_final_labeled_space()
    # x.play(path=path, videowriter=False)

    with open(os.path.join(network_dir, 'time_series_selected_states_new.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    with open(os.path.join(network_dir, 'state_series_selected_states_new.json'), 'r') as json_file:
        state_series_dict = json.load(json_file)
        json_file.close()

    del time_series_dict['S_SPT_4350024_SSpecialT_1_ants']
    del state_series_dict['S_SPT_4350024_SSpecialT_1_ants']

    time_series_dict['S_SPT_4350024_SSpecialT_1_ants'] = path.time_series
    state_series_dict['S_SPT_4350024_SSpecialT_1_ants'] = path.state_series

    # df_all = pd.concat([pd.concat(dfs_ant_old.values()), df_relevant])
    # time_series_dict, state_series_dict = Path.create_dicts(df_all, ConfigSpace_SelectedStates,
    #                                                         dictio_ts=time_series_dict, dictio_ss=state_series_dict)
    Path.save_dicts(time_series_dict, state_series_dict, name='_selected_states_new')
    # DEBUG = 1

    # df_relevant['time series'] = df_relevant['filename'].map(time_series_dict)
    # df_relevant['state series'] = df_relevant['filename'].map(state_series_dict)

    # Path.plot_paths(zip(['ant', 'ant_old', 'human', 'pheidole'], [dfs_ant, dfs_ant_old, dfs_human, dfs_pheidole]))
    Path.plot_paths(zip(['ant_old'], [dfs_ant_old]), time_series_dict)

    # ConfigSpace_class = ConfigSpace_AdditionalStates
    # to_add = Path.find_missing(myDataFrame)
    # to_add = ['L_SPT_4660001_LSpecialT_1_ants (part 1)']
    # time_series_dict, state_series_dict = Path.add_to_dict(to_add,
    #                                                        ConfigSpace_class,
    #                                                        time_series_dict,
    #                                                        state_series_dict)
    # time_series_dict, state_series_dict = Path.create_edge_walked_dict(myDataFrame, ConfigSpace_class)

    # filenames = []
    # to_recalculate = myDataFrame[myDataFrame['filename'].isin(filenames)]
    # time_series_dict_selected_states, state_series_dict_selected_states = Path.add_to_dict(to_recalculate,
    #                                             ConfigSpace_class, time_series_dict_selected_states,
    #                                             state_series_dict_selected_states)

    # Path.save_dicts(time_series_dict, state_series_dict, name='_selected_states')
