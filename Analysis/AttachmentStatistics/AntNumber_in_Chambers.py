from trajectory_inheritance.get import get
from Setup.Maze import Maze
import plotly.graph_objects as go
from DataFrame.import_excel_dfs import *
from scipy.signal import medfilt
from Directories import network_dir
import os
import numpy as np
from ConfigSpace.ConfigSpace_SelectedStates import states
import pandas as pd
import json

ts_step = 0.25  # time step in seconds of time series
chamberCount_fps = 1  # number of frames per second in chamber count time series

removed_states = ['g', 'eb', 'i']
combined_states = {'a': ['ab', 'ac'], 'b_stuck': ['b1', 'b2', 'be'], 'b': ['b'], 'c': ['c', 'cg'], 'e': ['e', 'eg'],
                   'f': ['f'], 'h': ['h']}

states = [state for state in states if state not in removed_states]


class ChamberCounter:
    def __init__(self, x, ts=None):
        self.x = x
        self.ts = ts[::int(chamberCount_fps / ts_step)] + [ts[-1]]
        self.raw_counts = None
        self.filtered_counts = None
        self.windowSize = 5
        self.state_counts = None

    def __add__(self, other):
        if self.filtered_counts is None:
            self.calc_filtered_counts()
        if other.filtered_counts is None:
            other.calc_filtered_counts()

        self.filtered_counts = {chamber: self.filtered_counts[chamber] + other.filtered_counts[chamber]
                                for chamber in range(3)}
        self.state_counts = {state: self.state_counts[state] + other.state_counts[state]
                             for state in states}
        return self

    def calc_raw_counts(self, attached=True, non_attached=True):
        if self.x.participants is None:
            self.x.load_participants()

        m = Maze(self.x)
        self.raw_counts = {0: [], 1: [], 2: []}
        for frame_number, frame in zip(self.x.frames, self.x.participants.frames):

            if frame_number == 8620:
                DEBUG = 1
            ant_locations = []

            if attached:
                ant_locations += frame.position[frame.carrying.astype(bool)].tolist()

            if non_attached:
                ant_locations += frame.position[~frame.carrying.astype(bool)].tolist()

            chamber_ints = [m.in_what_Chamber(r) for r in ant_locations]
            [self.raw_counts[i].append(chamber_ints.count(i)) for i in range(3)]

    def calc_filtered_counts(self):
        if self.raw_counts is None:
            self.calc_raw_counts()
        self.filtered_counts = {chamber: medfilt(count, self.windowSize).tolist()
                                for chamber, count in self.raw_counts.items()}

    def calc_state_counts(self):
        if self.filtered_counts is None:
            self.calc_filtered_counts()
        states_in_exp = [state for state in self.ts]

        print(len(states_in_exp))
        print(len(self.filtered_counts[0]))
        # if abs((len(states_in_exp)/len(self.filtered_counts[0]) - 1)) > 0.0007:
        # raise ValueError('Number of states and number of frames... do they match?')

        self.state_counts = {state: [] for state in states}

        for state, counts in zip(states_in_exp, zip(*self.filtered_counts.values())):
            if state not in removed_states:
                self.state_counts[state].append(counts)

    def get_number_of_ants_based_on_CM(self):
        if self.filtered_counts is None:
            self.calc_filtered_counts()

        maze = Maze(self.x)
        chamber_sizes = {0: maze.arena_height * maze.slits[0],
                         1: maze.arena_height * (maze.slits[1] - maze.slits[0]),
                         2: maze.arena_height * (maze.slits[1] - maze.slits[0])}

        in_0 = self.x.position[:, 0] < maze.slits[0]
        in_1 = np.logical_and(self.x.position[:, 0] > maze.slits[0], self.x.position[:, 0] < maze.slits[1])

        CM_in_chamber = np.zeros_like(self.x.position[:, 0])
        CM_in_chamber[in_0] = 0
        CM_in_chamber[in_1] = 1

        CM_inChamber_counts = {state: [] for state in [0, 1]}

        if abs(len(CM_in_chamber)-len(self.filtered_counts[2])) > 1:
            raise ValueError('Number of frames in CM_in_chamber and number of frames in filtered_counts do not match')

        for chamber, counts in zip(CM_in_chamber, zip(*self.filtered_counts.values())):
            CM_inChamber_counts[chamber].append(list(counts))

        CM_inChamber_counts_mean = {chamber: np.mean(counts, axis=0) for chamber, counts in CM_inChamber_counts.items()
                                    if len(counts) > 0}
        CM_inChamber_counts_density = {chamber: np.divide(counts, list(chamber_sizes.values())).tolist() for
                                       chamber, counts
                                       in CM_inChamber_counts_mean.items()}

        return pd.Series(CM_inChamber_counts_density)

    def plot(self, y: dict, fig=None):
        if fig is None:
            fig = go.Figure()
        var = [fig.add_traces([go.Scatter(x=list(range(len(self.x.frames))),
                                          y=y[i],
                                          mode='lines', name=f'Chamber {i}') for i in range(3)])]
        fig.update_layout(xaxis=dict(tickmode='array',
                                     tickvals=list(range(len(self.x.frames)))[::100],
                                     ticktext=list(self.x.frames)[::100]))

    def plot_state_counts(self, mean_state_counts, std_state_counts, fig=None):
        if fig is None:
            fig = go.Figure()

        max_count = 10

        for i, color in zip([2, 1, 0], ['#CB4335', '#16A085', '#EB89B5']):
            ch = [mean_state_counts[state][i] for state in mean_state_counts.keys()]
            max_count = max(max_count, np.nanmax(np.array(ch)).astype(int) + 10)
            std = [std_state_counts[state][i] for state in std_state_counts.keys()]
            fig.add_trace(go.Bar(x=ch, name='chamber' + str(i), opacity=0.75, error_x=dict(type='data', array=std),
                                 marker_color=color))

        fig.update_yaxes(
            ticktext=[s for s in std_state_counts.keys()],
            tickvals=[i for i in range(len(std_state_counts.keys()))],
        )

        fig.update_xaxes(
            ticktext=list(range(0, max_count, 10)),
            tickvals=list(range(0, max_count, 10)),
        )

        fig.update_layout(
            # title_text=size,  # title of plot
            xaxis_title_text='count: ' + size,  # xaxis label
            yaxis_title_text='state',  # yaxis label
            bargap=0.2,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1,  # gap between bars of the same location coordinates
            legend=dict(x=0.5, y=0.99, font=dict(size=10, family="Times New Romans"), orientation='h'),
            paper_bgcolor='rgba(0,0,0,0)',
        )

    def save(self):
        if self.filtered_counts is None:
            self.calc_filtered_counts()
        with open(os.path.join('Chamber_Counts', f'{self.x.filename}_chamber_counts.json'), 'w') as f:
            json.dump(self.filtered_counts, f)
        print('saved in ' + os.path.join('Chamber_Counts', f'{self.x.filename}_chamber_counts.json'))

        if self.state_counts is None:
            self.calc_state_counts()
        with open(os.path.join('Chamber_Counts', f'{self.x.filename}_state_counts.json'), 'w') as f:
            json.dump(self.state_counts, f)
        print('saved in ' + os.path.join('Chamber_Counts', f'{self.x.filename}_state_counts.json'))

        # if self.state_counts is None:
        #     self.calc_state_counts()
        # with open(os.path.join('Chamber_Counts', f'{self.x.filename}_state_counts.json'), 'w') as f:
        #     json.dump(self.state_counts, f)
        # print('saved in ' + os.path.join('Chamber_Counts', f'{self.x.filename}_state_counts.json'))

    def load(self):
        with open(os.path.join('Chamber_Counts\\without_attached', f'{self.x.filename}_chamber_counts.json'), 'r') as f:
            self.filtered_counts = json.load(f)
            # transform keys from str to int
            self.filtered_counts = {int(k): v for k, v in self.filtered_counts.items()}

        with open(os.path.join('Chamber_Counts\\without_attached', f'{self.x.filename}_state_counts.json'), 'r') as f:
            self.state_counts = json.load(f)

    def empty(self):
        self.raw_counts = {chamber: [] for chamber in range(3)}
        self.filtered_counts = {chamber: [] for chamber in range(3)}
        self.state_counts = {state: [] for state in states}

    @staticmethod
    def calculate_fractions(state_counts):
        combined_state_counts = {superstate: [el for s in substates for el in state_counts[s]] \
                                 for superstate, substates in combined_states.items()}

        mean_state_counts = {superstate: np.array(counts).mean(axis=0) \
            if len(counts) else [np.NAN, np.NAN, np.NAN]
                             for superstate, counts in combined_state_counts.items()}
        std_state_counts = {superstate: np.array(counts).std(axis=0) \
            if len(counts) else [np.NAN, np.NAN, np.NAN]
                            for superstate, counts in combined_state_counts.items()}
        return mean_state_counts, std_state_counts


# with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
#     time_series_dict = json.load(json_file)
#     json_file.close()

df_exps = pd.read_excel('decent_tracking.xlsx', engine='openpyxl').dropna(subset=['size'])
groups = df_exps.groupby('size')


def mean_one_size(df) -> dict:
    d = {}
    for filename in df['filename']:
        x = get(filename)
        x.adapt_fps(1)
        cc = ChamberCounter(x)
        cc.load()
        d[filename] = cc.get_number_of_ants_based_on_CM()
    # keys is the chamber that the CenterofMass is in
    div_by_chamber = {chamber: [d[filename][chamber] for filename in df['filename'] if chamber in d[filename].keys()]
                      for chamber in range(2)}

    for ch, counts in div_by_chamber.items():
        if len(counts) == 0:
            div_by_chamber[ch] = [np.nan, np.nan, np.nan]
        else:
            div_by_chamber[ch] = np.array(counts).mean(axis=0)
    return div_by_chamber


if __name__ == '__main__':

    # filename = 'S_SPT_4770007_SSpecialT_1_ants (part 1)'
    # x = get(filename)
    # x.adapt_fps(chamberCount_fps * x.fps)
    # cc = ChamberCounter(x)
    # cc.get_number_of_ants_based_on_CM()
    # x.play(10)
    # #
    # cc.calc_state_counts()

    # do it only for 'L'

    CM_density = {}
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=4)

    col_dict = {'XL': 4, 'L': 3, 'M': 2, 'S': 1}
    colors = {0: 'red', 1: 'blue'}
    for size, df in groups:
        CM_density[size] = mean_one_size(df)
        for CM_ch in [0, 1]:
            fig = fig.add_scatter(
                x=[0, 1, 2],
                y=CM_density[size][CM_ch],
                name=size + ' ch' + str(CM_ch),
                # color=colors[CM_ch],
                row=1, col=col_dict[size])
            # set color of the line
            fig.data[-1].line.color = colors[CM_ch]
        # change y-axis range to [0, 2]
        fig.update_yaxes(range=[0, 6], row=1, col=col_dict[size])

        # set title of each subplot
        # fig.update_layout(title_text=f'{size} ants', row=1, col=i)

    for size, df in groups:
        for filename in df['filename']:
            if not path.isfile(os.path.join('Chamber_Counts\\without_attached', f'{filename}_chamber_counts.json')):
                x = get(filename)
                x.adapt_fps(1)
                cc = ChamberCounter(x)
                cc.calc_raw_counts(with_attached=False)
                cc.calc_filtered_counts()

                fig = go.Figure()
                cc.plot(y=cc.filtered_counts, fig=fig)
                cc.plot(y=cc.raw_counts, fig=fig)
                fig.write_image(os.path.join('Chamber_Counts', f'{filename}_chamber_counts.png'),
                                width=1400, height=700, scale=2)
                cc.calc_state_counts()
                cc.save()

    for size, df in groups:
        cc_mean = ChamberCounter(None)
        cc_mean.empty()

        for filename in df['filename']:
            x = get(filename)
            cc = ChamberCounter(x)
            cc.load()
            cc_mean = cc_mean + cc

        fig = go.Figure()
        cc_mean.plot_state_counts(*cc_mean.calculate_fractions(cc_mean.state_counts), fig=fig)
        fig.write_image(os.path.join('Chamber_Counts', f'{size}_state_counts.png'),
                        width=700, height=700, scale=2)
