import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from DataFrame.plot_dataframe import save_fig
from trajectory_inheritance.get import get
import os
import json
from tqdm import tqdm
from Directories import network_dir
from trajectory_inheritance.trajectory import Trajectory_part
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from DataFrame.import_excel_dfs import dfs_ant, df_minimal, dfs_ant_old
from DataFrame.gillespie_dataFrame import dfs_gillespie
from plotly import express as px
from ConfigSpace.experiment_sliding import Experiment_Sliding
from Analysis.greed import Trajectory
import pandas as pd

#
# class Traj_in_cg(Trajectory_part):
#
#     def __init__(self):
#         super().__init__(parent_traj, VideoChain, frames, tracked_frames, states=None)


def extend_time_series_to_match_frames(ts, traj):
    indices_to_ts_to_frames = np.cumsum([1 / (int(len(traj.frames) / len(ts) * 10) / 10)
                                         for _ in range(len(traj.frames))]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def cut_traj(traj, ts, buffer=0) -> tuple:
    """
    Divide the trajectory into two lists.
    First, cut off the exp after the last c state.
    Second, split traj into subtrajs that are (1) outside c and cg and (2) within c and cg.
    Return these two lists.
    """
    ts_extended = extend_time_series_to_match_frames(ts, traj)
    indices_cg = np.where(np.logical_or(np.array(ts_extended) == 'c', np.array(ts_extended) == 'cg'))[0]
    ind_succ = np.split(indices_cg, np.where(np.diff(indices_cg) != 1)[0] + 1)

    in_cg = np.zeros_like(traj.frames).astype(bool)
    in_cg[np.hstack(ind_succ)] = True

    where_in_cg = np.array([i for i, x in enumerate(in_cg) if x])
    index_successions_in_cg = np.split(where_in_cg, np.where(np.diff(where_in_cg) != 1)[0] + 1)

    where_out_cg = np.array([i for i, x in enumerate(in_cg) if not x])
    index_successions_out_cg = np.split(where_out_cg, np.where(np.diff(where_out_cg) != 1)[0] + 1)

    in_c_trajs = [Trajectory_part(traj, frames=[i[0]-buffer*traj.fps, i[-1]+buffer*traj.fps], VideoChain=[],
                                  tracked_frames=[], states=ts_extended)
                  for i in index_successions_in_cg if len(i) > 0]
    out_c_trajs = [Trajectory_part(traj, frames=[i[0]-buffer*traj.fps, i[-1]+buffer*traj.fps], VideoChain=[],
                                   tracked_frames=[], states=ts_extended)
                   for i in index_successions_out_cg if len(i) > 0]
    return in_c_trajs, out_c_trajs


def cut_traj_after_c_e_crossing(traj) -> Trajectory_part:
    """
    Divide the trajectory into two lists.
    First, cut off the exp after the last c state.
    Second, split traj into subtrajs that are (1) outside c and cg and (2) within c and cg.
    Return these two lists.
    """
    ts_extended = extend_time_series_to_match_frames(time_series_dict[traj.filename], traj)
    indices_e = np.where(np.array(ts_extended) == 'e')[0]
    return Trajectory_part(traj, frames=[indices_e[0], -1], VideoChain=[], tracked_frames=[])


def pL(traj):
    return PathLength(traj).translational_distance(smooth=False) + PathLength(traj).rotational_distance(smooth=False)


def after_exited(self, state: str, nth: int) -> list:
    """
    Index when state was exited for the nth time.
    """
    in_state = (np.array(['0'] + self.time_series + ['0']) == state).astype(int)
    entered_exited = np.where(in_state[:-1] != in_state[1:])[0]

    if entered_exited.size == 0:
        return []

    times = np.split(entered_exited, int(len(entered_exited) / 2))
    # time_series[0:15] = 'ab1'
    if len(times) <= nth:
        return []
    if len(times[nth]) < 2:
        raise Exception()
    return self.time_series[times[nth][1]:]


def c_to_e_passage_func(filename, ts) -> list:
    t = Trajectory(filename, time_series=ts)

    c_ac_or_e = {nth_time: t.state1_before_state2('ac', 'e', time_series=t.after_exited(['c', 'cg'], nth_time))
                 for nth_time in range(100)}
    c_ac_or_e = {key: value for key, value in c_ac_or_e.items() if value is not None}
    return list(c_ac_or_e.values())

# THIS is not in In_the_bottle class
# def distance_on_off_edge(in_c_list, ps, radius, edge_walk=None) -> tuple:
#     on_edge, off_edge = [], []
#
#     for traj in in_c_list:
#         if edge_walk is None:
#             edge_walk = Experiment_Sliding(traj)
#         else:
#             edge_walk.new_traj(traj)
#         edge_walk.find_on_edge(ps, radius)
#         edge_walk.clean_exp_from_short_values()
#         ind_succ = Experiment_Sliding.index_successions(edge_walk.on_edge)
#
#         edge_boolean = np.zeros_like(traj.frames).astype(bool)
#         edge_boolean[np.hstack(ind_succ)] = True
#
#         where_on_edge = np.array([i for i, x in enumerate(edge_boolean) if x])
#         index_successions_on_edge = np.split(where_on_edge, np.where(np.diff(where_on_edge) != 1)[0] + 1)
#
#         where_off_edge = np.array([i for i, x in enumerate(edge_boolean) if not x])
#         index_successions_off_edge = np.split(where_off_edge, np.where(np.diff(where_off_edge) != 1)[0] + 1)
#
#         on_edge += [pL(Trajectory_part(traj, frames=[i[0], i[-1]], VideoChain=[], tracked_frames=[]))
#                     for i in index_successions_on_edge if len(i) > 0 and i[-1] - i[0] > 0]
#         off_edge += [pL(Trajectory_part(traj, frames=[i[0], i[-1]], VideoChain=[], tracked_frames=[]))
#                      for i in index_successions_off_edge if len(i) > 0 and i[-1] - i[0] > 0]
#     return on_edge, off_edge


def calc_and_save_all(radius):
    df_shift = pd.read_excel('results\\df_shift.xlsx')
    for size, df_size in df_dict_sep_by_size.items():
        ps = ConfigSpace_SelectedStates(solver='ant', size=df_size['size'].iloc[0], shape='SPT',
                                        geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                                  'LoadDimensions_new2021_SPT_ant.xlsx'))
        ew = None
        filenames_to_calc = [filename for filename in df_size['filename'] if filename not in pL_in_c.keys()]
        for filename in tqdm(filenames_to_calc):
            # filename = 'XL_SPT_4290004_XLSpecialT_1_ants (part 1)' # M_SPT_4700003_MSpecialT_1_ants (part 1)
            traj = get(filename)

            # if traj.fps < 30:
            #     traj.adapt_fps(30)

            if traj.solver == 'ant' and \
                traj.geometry() != ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'):
                traj_new = traj.confine_to_new_dimensions()
            else:
                traj_new = traj

            traj_new.smooth()

            # traj.play()
            if ew is None:
                ew = Experiment_Sliding(traj_new)
            print('===========================')
            print(filename, ' starting')

            if 'c' in time_series_dict[filename]:
                in_cs, out_cs = cut_traj(traj_new, time_series_dict[filename])

                for in_c in in_cs:
                    if filename in df_shift['filename'].values:
                        shift_x = df_shift[(df_shift['filename'] == filename) &
                                           (df_shift['start_frame'] == in_c.frames_of_parent[0])]['shift_x'].values[0]
                        shift_y = df_shift[(df_shift['filename'] == filename) &
                                           (df_shift['start_frame'] == in_c.frames_of_parent[0])]['shift_y'].values[0]
                        in_c.position = in_c.position - np.array([shift_x, shift_y])
                    # in_c.play(geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

                times_entered_to_c[filename] = len(in_cs)
                pL_in_c[filename] = [pL(traj) for traj in in_cs]
                pL_out_c[filename] = [pL(traj) for traj in out_cs[:-1]]  # before they finally passed the bottleneck
                distance_in_c_on_edge[filename], distance_in_c_off_edge[filename] = \
                    distance_on_off_edge(in_cs, ps, radius, edge_walk=ew)
                print(sum(distance_in_c_on_edge[filename])/sum(distance_in_c_on_edge[filename] + distance_in_c_off_edge[filename]))
                c_to_e_passage[filename] = c_to_e_passage_func(filename, time_series_dict[filename])

            else:
                pL_in_c[filename] = []
                pL_out_c[filename] = []
                distance_in_c_on_edge[filename] = []
                distance_in_c_off_edge[filename] = []
                c_to_e_passage[filename] = []

            if 'e' in time_series_dict[filename]:
                after_e = cut_traj_after_c_e_crossing(traj)
                pL_after_first_e[filename] = pL(after_e)
            else:
                pL_after_first_e[filename] = None
            print(filename, ' ending')
            DEBUG = 1

    with open('results\\pL_in_c' + _add + '.json', 'w') as f:
        json.dump(pL_in_c, f)
    with open('results\\pL_out_c' + _add + '.json', 'w') as f:
        json.dump(pL_out_c, f)
    with open('results\\pL_after_first_e' + _add + '.json', 'w') as f:
        json.dump(pL_after_first_e, f)
    with open('results\\times_entered_to_c' + _add + '.json', 'w') as f:
        json.dump(times_entered_to_c, f)
    with open('results\\distance_in_c_on_edge' + _add + '.json', 'w') as f:
        json.dump(distance_in_c_on_edge, f)
    with open('results\\distance_in_c_off_edge' + _add + '.json', 'w') as f:
        json.dump(distance_in_c_off_edge, f)
    with open('results\\c_to_e_passage' + _add + '.json', 'w') as f:
        json.dump(c_to_e_passage, f)


def plot_in_CS_and_save(df_plot):
    for group_size, df in df_plot.groupby('size'):
        ps = ConfigSpace_SelectedStates(solver=solver, size=df.iloc[0]['size'], shape=shape, geometry=(
            'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
        ps.load_final_labeled_space()
        ew = None

        # create a pandas dataframe with columns: 'filename', 'start_frame', 'end_frame', 'shift_x', 'shift_y'
        # df_shift = pd.DataFrame(columns=['filename', 'start_frame', 'end_frame', 'shift_x', 'shift_y'])
        df_shift = pd.read_excel('results\\df_shift.xlsx')

        for filename, on_edge_fraction in zip(df['filename'], df['on_edge_fraction']):
            print(filename)
            print(on_edge_fraction)
            traj = get(filename)

            if traj.fps < 30:
                traj.adapt_fps(30)

            if traj.solver == 'ant' and \
                traj.geometry() != ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'):
                traj_new = traj.confine_to_new_dimensions()
            else:
                traj_new = traj

            in_cs, out_cs = cut_traj(traj_new, time_series_dict[filename])
            # out_cs[0].play(geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
            # traj.position = traj.position + np.array([0.2, 0])

            for in_c in in_cs:
                if filename in df_shift['filename'].values:
                    shift_x = df_shift[(df_shift['filename'] == filename) &
                                       (df_shift['start_frame'] == in_c.frames_of_parent[0])]['shift_x'].values[0]
                    shift_y = df_shift[(df_shift['filename'] == filename) &
                                       (df_shift['start_frame'] == in_c.frames_of_parent[0])]['shift_y'].values[0]
                    in_c.position = in_c.position - np.array([shift_x, shift_y])
                in_c.play(geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
                if ew is None:
                    ew = Experiment_Sliding(in_c)
                    ew.find_boundary(ps=ps, radius=radius)
                else:
                    ew.new_traj(in_c)
                ew.find_on_edge(ps)

                Experiment_Sliding.plot_traj_in_cs(ps=ps, traj=in_c, bool_to_plot=ew.on_edge)

                # shift_x, shift_y = 0, 0
                # append to the dataframe df_shift the filename, start_frame, end_frame, shift_x, shift_y
                # df_shift = df_shift.append({'filename': filename,
                #                             'start_frame': in_c.frames_of_parent[0],
                #                             'end_frame': in_c.frames_of_parent[-1],
                #                             'shift_x': shift_x, 'shift_y': shift_y},
                #                            ignore_index=True)
            # save df_shift to excel file
        DEBUG = 1


def plot_correlations(df):
    df['pL_in_c'] = df['filename'].map(pL_in_c)
    # drop all the columns which have an empty list in the pL_in_c column
    df = df[df['pL_in_c'].apply(len) > 0]

    df['distance_in_c_on_edge'] = df['filename'].map(distance_in_c_on_edge).apply(np.sum)
    df['distance_in_c_off_edge'] = df['filename'].map(distance_in_c_off_edge).apply(np.sum)
    df['on_edge_fraction'] = df['distance_in_c_on_edge'] / (
            df['distance_in_c_on_edge'] + df['distance_in_c_off_edge'])
    df['c_to_e_passage'] = df['filename'].map(c_to_e_passage)
    df['prob'] = df['c_to_e_passage'].apply(np.mean)

    # plot pL_in_c vs. on_edge_fraction with px
    fig = px.scatter(df, x='on_edge_fraction', y='prob', color='size', trendline='ols')
    fig.update_layout(title='pL_in_c vs. on_edge_fraction')
    fig.show()



    DEBUG = 1


def plot_means():
    median_pL_in_c_size, std_pL_in_c_size = {}, {}
    median_pL_out_c_size, std_pL_out_c_size = {}, {}
    median_times_entered_to_c, std_times_entered_to_c = {}, {}
    median_distance_in_c_on_edge, std_distance_in_c_on_edge = {}, {}
    median_distance_in_c_off_edge, std_distance_in_c_off_edge = {}, {}
    median_on_edge_fraction, std_on_edge_fraction = {}, {}
    median_pL_after_first_e, std_pL_after_first_e = {}, {}
    median_c_to_e_passage, std_c_to_e_passage = {}, {}

    for size, df in df_dict_sep_by_size.items():
        print(size)
        if size == 'S':
            DEBUG = 1
        minimal = df_minimal[(df_minimal['size'] == df.iloc[0]['size']) & (df_minimal['shape'] == 'SPT') & (
                df_minimal['initial condition'] == 'back')].iloc[0]['path length [length unit]']

        df['pL_in_c'] = df['filename'].map(pL_in_c)
        df['pL_out_c'] = df['filename'].map(pL_out_c)
        df['pL_after_first_e'] = df['filename'].map(pL_after_first_e)
        df['times_entered_to_c'] = df['filename'].map(times_entered_to_c)
        df['distance_in_c_on_edge'] = df['filename'].map(distance_in_c_on_edge).apply(np.sum)
        df['distance_in_c_off_edge'] = df['filename'].map(distance_in_c_off_edge).apply(np.sum)
        df['c_to_e_passage'] = df['filename'].map(c_to_e_passage)

        pL_in_c_size = [item / minimal for sublist in df['pL_in_c'].dropna() for item in sublist]
        median_pL_in_c_size[size] = np.mean(pL_in_c_size)
        std_pL_in_c_size[size] = np.std(pL_in_c_size)

        pL_out_c_size = [item / minimal for sublist in df['pL_out_c'].dropna() for item in sublist]
        median_pL_out_c_size[size] = np.mean(pL_out_c_size)
        std_pL_out_c_size[size] = np.std(pL_out_c_size)

        pL_after_first_e_size = [item / minimal for item in df['pL_after_first_e'].dropna()]
        median_pL_after_first_e[size] = np.mean(pL_after_first_e_size)
        std_pL_after_first_e[size] = np.std(pL_after_first_e_size)

        times_entered_to_c_size = df['times_entered_to_c'].dropna().tolist()
        median_times_entered_to_c[size] = np.mean(times_entered_to_c_size)
        std_times_entered_to_c[size] = np.std(times_entered_to_c_size)

        distance_in_c_on_edge_size = list(df['distance_in_c_on_edge'].dropna() / minimal)
        median_distance_in_c_on_edge[size] = np.mean(distance_in_c_on_edge_size)
        std_distance_in_c_on_edge[size] = np.std(distance_in_c_on_edge_size)

        distance_in_c_off_edge_size = list(df['distance_in_c_off_edge'].dropna() / minimal)
        median_distance_in_c_off_edge[size] = np.mean(distance_in_c_off_edge_size)
        std_distance_in_c_off_edge[size] = np.std(distance_in_c_off_edge_size)

        df['on_edge_fraction'] = df['distance_in_c_on_edge'] / (
                df['distance_in_c_on_edge'] + df['distance_in_c_off_edge'])
        on_edge_fraction_size = df['on_edge_fraction'].dropna().tolist()
        median_on_edge_fraction[size] = np.mean(on_edge_fraction_size)
        std_on_edge_fraction[size] = np.std(on_edge_fraction_size)

        c_to_e_passage_size = [not a for b in list(df['c_to_e_passage'].dropna()) for a in b]
        median_c_to_e_passage[size] = np.mean(c_to_e_passage_size)
        std_c_to_e_passage[size] = np.std(c_to_e_passage_size)

    fig_pL_in_c = px.bar(x=list(median_pL_in_c_size.keys()),
                         y=list(median_pL_in_c_size.values()),
                         error_y=list(std_pL_in_c_size.values()))
    fig_pL_in_c.update_yaxes(title_text='mean pL in c/minimal pL')
    fig_pL_in_c.update_xaxes(title_text='size')
    save_fig(fig_pL_in_c, 'pL_in_c' + _add)

    fig_pL_out_c = px.bar(x=list(median_pL_out_c_size.keys()),
                          y=list(median_pL_out_c_size.values()),
                          error_y=list(std_pL_out_c_size.values()))
    fig_pL_out_c.update_yaxes(title_text='mean pL outside of  c/minimal pL')
    fig_pL_out_c.update_xaxes(title_text='size')
    save_fig(fig_pL_out_c, 'pL_out_c' + _add)

    fig_pL_after_first_e = px.bar(x=list(median_pL_after_first_e.keys()),
                                  y=list(median_pL_after_first_e.values()),
                                  error_y=list(std_pL_out_c_size.values()))
    fig_pL_after_first_e.update_yaxes(title_text='mean_pL_after_first_e/minimal pL')
    fig_pL_after_first_e.update_xaxes(title_text='size')
    save_fig(fig_pL_after_first_e, 'pL_after_first_e' + _add)

    fig_times_entered_to_c = px.bar(x=list(median_times_entered_to_c.keys()),
                                    y=list(median_times_entered_to_c.values()),
                                    error_y=list(std_times_entered_to_c.values()))
    fig_times_entered_to_c.update_yaxes(title_text='times entered to c per exp')
    fig_times_entered_to_c.update_xaxes(title_text='size')
    save_fig(fig_times_entered_to_c, 'times_entered_to_c' + _add)

    fig_distance_in_c_on_edge = px.bar(x=list(median_distance_in_c_on_edge.keys()),
                                       y=list(median_distance_in_c_on_edge.values()),
                                       error_y=list(std_distance_in_c_on_edge.values()))
    fig_distance_in_c_on_edge.update_yaxes(title_text='pL_in_c_on_edge/minimal pL')
    fig_distance_in_c_on_edge.update_xaxes(title_text='size')
    save_fig(fig_distance_in_c_on_edge, 'pL_in_c_on_edge' + _add)

    fig_distance_in_c_off_edge = px.bar(x=list(median_distance_in_c_off_edge.keys()),
                                        y=list(median_distance_in_c_off_edge.values()),
                                        error_y=list(std_distance_in_c_off_edge.values()))
    fig_distance_in_c_off_edge.update_yaxes(title_text='pL_in_c_off_edge/minimal pL')
    fig_distance_in_c_off_edge.update_xaxes(title_text='size')
    save_fig(fig_distance_in_c_off_edge, 'pL_in_c_off_edge' + _add)

    fig_on_edge_fraction = px.bar(x=list(median_on_edge_fraction.keys()),
                                  y=list(median_on_edge_fraction.values()),
                                  error_y=list(std_on_edge_fraction.values()))
    fig_on_edge_fraction.update_yaxes(title_text='mean_on_edge_fraction')
    fig_on_edge_fraction.update_xaxes(title_text='size')
    save_fig(fig_on_edge_fraction, 'on_edge_fraction' + _add)

    fig_c_to_e_passage = px.bar(x=list(median_c_to_e_passage.keys()),
                                y=list(median_c_to_e_passage.values()),
                                error_y=list(std_c_to_e_passage.values()))
    fig_c_to_e_passage.update_yaxes(title_text='prob to pass c to e')
    fig_c_to_e_passage.update_xaxes(title_text='size')
    save_fig(fig_c_to_e_passage, 'c_to_e_passage' + _add)


# with open(os.path.join('C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\ConfigSpace\\results\\') + \
# 'edge_walk.json', 'r') as f:
#     edge_walks = json.load(f)


if __name__ == '__main__':
    solver, shape = 'gillespie', 'SPT'
    df_dict_sep_by_size = dfs_gillespie
    # x = get(df_dict_sep_by_size['S'].iloc[0]['filename'])
    # x.play()
    radius = 3
    _add = '_gillespie' + str(radius)
    with open(os.path.join(network_dir, 'time_series_selected_states_gillespie' + '.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # ========================== Statistics ========================

    filename = 'sim_L_20230128-020528New'
    traj = get(filename)
    ps = ConfigSpace_SelectedStates(solver=traj.solver, size=traj.size, shape=traj.shape,
                                    geometry=traj.geometry())

    if 'c' in time_series_dict[filename]:
        in_c, out_c = cut_traj(traj, time_series_dict[filename])
        print('times_entered_to_c', len(in_c))
        print('pL_in_c', [pL(traj) for traj in in_c])
        print('pL_out_c', [pL(traj) for traj in out_c[:-1]])  # this is before they finally passed the bottleneck
        on, off = distance_on_off_edge(in_c, ps, radius=radius)
        print('distance_on_edge',  on)
        print('distance_off_edge',  off)
        print('on_edge_fraction', sum(on)/sum(on+off))

    if 'e' in time_series_dict[filename]:
        after_e = cut_traj_after_c_e_crossing(traj)
        print('pL_after_first_e', pL(after_e))

    DEBUG = 1

    # solver, shape = 'ant', 'SPT'
    # # df_dict_sep_by_size = dfs_ant
    # df = pd.concat(list(dfs_ant.values()) + list(dfs_ant_old.values()))
    # df_dict_sep_by_size = {size: pd.concat([dfs_ant[size], dfs_ant_old[size]]) for size in dfs_ant.keys()}
    # # del df_dict_sep_by_size['XL']
    # # del df_dict_sep_by_size['L']
    # # del df_dict_sep_by_size['M']
    # # del df_dict_sep_by_size['S (> 1)']
    # # del df_dict_sep_by_size['Single (1)']

    # radius = 10
    # _add = '_ant' + str(radius)
    # with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    #     time_series_dict = json.load(json_file)
    #     json_file.close()

    # pL_in_c = {}
    # pL_out_c = {}
    # pL_after_first_e = {}
    # times_entered_to_c = {}
    # distance_in_c_on_edge = {}
    # distance_in_c_off_edge = {}
    # c_to_e_passage = {}

    with open('results\\pL_in_c' + _add + '.json', 'r') as f:
        pL_in_c = json.load(f)
    with open('results\\pL_out_c' + _add + '.json', 'r') as f:
        pL_out_c = json.load(f)
    with open('results\\pL_after_first_e' + _add + '.json', 'r') as f:
        pL_after_first_e = json.load(f)
    with open('results\\times_entered_to_c' + _add + '.json', 'r') as f:
        times_entered_to_c = json.load(f)
    with open('results\\distance_in_c_on_edge' + _add + '.json', 'r') as f:
        distance_in_c_on_edge = json.load(f)
    with open('results\\distance_in_c_off_edge' + _add + '.json', 'r') as f:
        distance_in_c_off_edge = json.load(f)
    with open('results\\c_to_e_passage' + _add + '.json', 'r') as f:
        c_to_e_passage = json.load(f)

    df = df_dict_sep_by_size['S']
    df['distance_in_c_on_edge'] = df['filename'].map(distance_in_c_on_edge).apply(np.sum)
    df['distance_in_c_off_edge'] = df['filename'].map(distance_in_c_off_edge).apply(np.sum)
    df['on_edge_fraction'] = df['distance_in_c_on_edge'] / (
            df['distance_in_c_on_edge'] + df['distance_in_c_off_edge'])
    #
    check = df[['filename', 'size', 'on_edge_fraction']]
    # check_XL = check[check['filename'] == 'XL']
    to_plot = check[(check['size'] == 'S') & (check['on_edge_fraction'] < 0.85)]['filename'].tolist()

    # filenames = ['XL_SPT_4290004_XLSpecialT_1_ants (part 1)', 'XL_SPT_4290006_XLSpecialT_1_ants',
    #              'XL_SPT_4290007_XLSpecialT_1_ants', 'XL_SPT_4290008_XLSpecialT_1_ants',
    #              'XL_SPT_4290008_XLSpecialT_2_ants',
    #              'XL_SPT_4290008_XLSpecialT_3_ants (part 1)', 'XL_SPT_4290009_XLSpecialT_2_ants',
    #              'XL_SPT_4290009_XLSpecialT_3_ants (part 1)', 'XL_SPT_4300003_XLSpecialT_1_ants',
    #              'XL_SPT_4300003_XLSpecialT_2_ants (part 1)', 'XL_SPT_4300004_XLSpecialT_2_ants',
    #              'XL_SPT_4300004_XLSpecialT_3_ants (part 1)', 'XL_SPT_4340011_XLSpecialT_1_ants (part 1)',
    #              'XL_SPT_4280005_XLSpecialT_1_ants (part 1)']

    df_plot = df[df['filename'].isin(to_plot)]
    plot_in_CS_and_save(df_plot)

    # calc_and_save_all(radius)
    # plot_means()
    # plot_correlations(df_dict_sep_by_size['M'])