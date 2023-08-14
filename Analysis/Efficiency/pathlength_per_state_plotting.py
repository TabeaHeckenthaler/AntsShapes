from Directories import home, network_dir
from matplotlib import pyplot as plt
import json
import numpy as np
from tqdm import tqdm
from colors import colors_humans as colors
import os
import pandas as pd
from DataFrame.import_excel_dfs import dfs_human

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# date = '2023_06_27'
date = 'SimTrjs_RemoveAntsNearWall=False'
date1 = 'SimTrjs_RemoveAntsNearWall=True'
#
df_gillespie = pd.read_excel(home + '\\Gillespie\\' + date + '_sim.xlsx')
df_gillespie1 = pd.read_excel(home + '\\Gillespie\\' + date1 + '_sim.xlsx')
df_human = pd.read_excel(home + '\\DataFrame\\final\\df_human.xlsx')
df_ant_excluded = pd.read_excel(home + '\\DataFrame\\final\\df_ant_excluded.xlsx')

states = ['ab', 'ac', 'b', 'be1', 'be2', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
# length by which translational distance is divided: Arena height
sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
                    'sim': ['S', 'M', 'L', 'XL'],
                    # 'sim': ['XS', 'S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium NC', 'Medium C', 'Large NC', 'Large C'],
                    }
scale_trans_per_solver = {'ant': {'Single (1)': 2.3875, 'S (> 1)': 2.3875, 'S': 2.3875,
                                  'M': 4.775, 'L': 9.55, 'XL': 19.1},
                          'sim': {'XS': 2.3875 / 2, 'S': 2.3875, 'M': 4.775, 'L': 9.55, 'XL': 19.1},
                          'human': {'Small': 3.3, 'Medium NC': 6.39, 'Medium C': 6.39, 'Large NC': 12.57,
                                    'Large C': 12.57,
                                    'Small Far': 3.3, 'Small Near': 3.3, 'Medium': 6.39, 'Large': 12.57},
                          }

# minimal_translation_ants = {'L': 12.73, 'M': 6.25, 'XL': 24.75, 'S (> 1)': 3.71, 'Single (1)': 3.71}
minimal_scaled_translation = 6.25 / 4.775 - 0.1

scale_rot_per_solver = {'ant': {'Single (1)': 2 * np.pi, 'S (> 1)': 2 * np.pi, 'S': 2 * np.pi, 'M': 2 * np.pi,
                                'L': 2 * np.pi, 'XL': 2 * np.pi},
                        'sim': {'XS': 2 * np.pi, 'S': 2 * np.pi, 'M': 2 * np.pi, 'L': 2 * np.pi, 'XL': 2 * np.pi},
                        'human': {'Small': 2 * np.pi, 'Medium NC': 2 * np.pi, 'Medium C': 2 * np.pi,
                                  'Large NC': 2 * np.pi,
                                  'Large C': 2 * np.pi,
                                  'Small Far': 2 * np.pi, 'Small Near': 2 * np.pi, 'Medium': 2 * np.pi,
                                  'Large': 2 * np.pi},
                        }

minimal_scaled_rotation = 1

solver_step = {'human': 0.05, 'ant': 1, 'sim': 1, 'pheidole': 1}

xlim_per_solver_trans = {'ant': [0, 100], 'sim': [0, 100], 'human': [0, 10]}
xlim_per_solver_rot = {'ant': [0, 100], 'sim': [0, 100], 'human': [0, 5]}


def find_size(filename):
    return filename.split('_')[0]


def translation_in_state(filename, state, json_file):
    return np.sum(list(json_file[filename][state][0].values()))


def rotation_in_state(filename, state, json_file):
    return np.sum(list(json_file[filename][state][1].values()))


def calc_trans_rot_per_state_per_filename(filenames, json_file):
    translation_in_states = {}
    rotation_in_states = {}
    for state in states:
        translation_in_states[state] = \
            {filename: translation_in_state(filename, state, json_file) for filename in filenames}
        rotation_in_states[state] = \
            {filename: rotation_in_state(filename, state, json_file) for filename in filenames}
    return translation_in_states, rotation_in_states


def get_size_groups(df, solver) -> dict:
    if solver in ['ant', 'sim', date1 + '_sim', date + '_sim']:
        # if only_winner:
        #     df = df[df['winner']]
        d = {size: group for size, group in df.groupby('size')}
        # if solver == 'ant':
        #     d.pop('Single (1)')
        # if solver == 'sim':
        #     d.pop('XS')
        return d

    elif solver == 'human':
        return dfs_human


def plot_trans_rot_in_histograms(df, translation_in_states, rotation_in_states):
    groups = get_size_groups(df, solver)

    trans_mean, rot_mean = {}, {}
    trans_std, rot_std = {}, {}

    for size, group in groups.items():
        filenames = group['filename'].tolist()
        trans_mean[size], rot_mean[size] = {}, {}
        trans_std[size], rot_std[size] = {}, {}
        for state in states:
            trans_mean[size][state] = np.mean([translation_in_states[state][filename] for filename in filenames])
            trans_std[size][state] = np.std([translation_in_states[state][filename]
                                             for filename in filenames]) / np.sqrt(len(filenames))
            rot_mean[size][state] = np.mean([rotation_in_states[state][filename] for filename in filenames])
            rot_std[size][state] = np.std([rotation_in_states[state][filename]
                                           for filename in filenames]) / np.sqrt(len(filenames))

    fig = plt.figure(figsize=(7, 5))
    for size in sizes:
        # it would be best to plot all into the same histogram
        plt.errorbar(states, [trans_mean[size][state] / scale_trans[size] for state in states],
                     yerr=[trans_std[size][state] / scale_trans[size] for state in states],
                     label=size, color=colors[size], linewidth=2, marker='*')
    plt.ylabel('translation / maze width')
    plt.legend()
    ylim = {'ant': 30, 'sim': 30, 'human': 1.5}
    plt.ylim([0, ylim[solver]])
    plt.title(solver_string)
    plt.tight_layout()
    plt.savefig('results\\' + solver_string + '_translation_per_state.png')
    plt.savefig('results\\' + solver_string + '_translation_per_state.svg')
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    for size in sizes:
        # it would be best to plot all into the same histogram
        plt.errorbar(states, [rot_mean[size][state] / (2 * np.pi) for state in states],
                     yerr=[rot_std[size][state] / (2 * np.pi) for state in states],
                     label=size, color=colors[size], linewidth=2, marker='*')
    plt.ylabel('rotation / 2pi')
    plt.legend()
    ylim = {'ant': 10, 'sim': 10, 'human': 0.4}
    plt.ylim([0, ylim[solver]])
    plt.title(solver_string)
    plt.tight_layout()
    plt.savefig('results\\' + solver_string + '_rotation_per_state.png')
    plt.savefig('results\\' + solver_string + '_rotation_per_state.svg')
    plt.close()


def measure_without_state(filename, exclude_state: list, measure):
    tr = 0
    for state in states:
        if state not in exclude_state:
            tr += measure[state][filename]
    return tr


def plot_CDF(df, measure, scale, xlim, minimal):
    df['measure'] = df['filename'].map(measure)

    dfs = get_size_groups(df, solver)

    fig = plt.figure(figsize=(7, 5))
    for size, df_size in tqdm(dfs.items()):
        df_individual = df[df['filename'].isin(df_size['filename'])]

        df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(scale)

        # sort by norm measure
        df_individual = df_individual.sort_values(by='norm measure')

        longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
        df_individual = df_individual[(df_individual['norm measure'] > longest_succ_experiment)
                                      | (df_individual['winner'])]
        x_values = np.arange(0, df_individual['norm measure'].max() + 2 * solver_step[solver], step=solver_step[solver])
        y_values = []

        for x in x_values:
            suc = df_individual[(df_individual['norm measure'] < x) & (df_individual['winner'])]
            y_values.append(len(suc) / len(df_individual))
        plt.step(x_values, y_values, label=size, color=colors[size], linewidth=2, where='post')

    # draw a vertical line at minimal_scaled_translation
    plt.axvline(minimal, color='black', linestyle='--', linewidth=2, alpha=0.5)
    # write 'minimal' next to the line vertically
    plt.text(minimal + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)

    plt.ylabel('% success')
    plt.ylim([0, 1])
    plt.xlim(xlim)
    plt.legend(prop={'size': 20})
    plt.title(solver_string)
    plt.tight_layout()


def calc_40th_percentile(df, measure, scale, solver):
    percentile_40th = {}
    df['measure'] = df['filename'].map(measure)
    dfs = get_size_groups(df, solver)
    for size, df_size in tqdm(dfs.items()):
        df_individual = df[df['filename'].isin(df_size['filename'])]

        df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(scale)

        # sort by norm measure
        df_individual = df_individual.sort_values(by='norm measure')

        longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
        df_individual = df_individual[(df_individual['norm measure'] > longest_succ_experiment)
                                      | (df_individual['winner'])]

        pLs = df_individual['norm measure'].copy()
        pLs[~df_individual['winner']] = np.inf

        percentile_40th[size] = np.percentile(pLs, 40)
    return percentile_40th


def calc_trans_rot_per_state_per_filename_measure_without_state():
    trans = \
        {filename: measure_without_state(filename, [], translation_in_states)
         for filename in filenames}

    with open(solver_string + '_trans.json', 'w') as f:
        json.dump(trans, f)

    rot = \
        {filename: measure_without_state(filename, [], rotation_in_states)
         for filename in filenames}

    with open(solver_string + '_rot.json', 'w') as f:
        json.dump(rot, f)


def finding_extremes():
    date1 = 'SimTrjs_RemoveAntsNearWall=False'
    date2 = 'SimTrjs_RemoveAntsNearWall=True'
    fig, ax = plt.subplots(figsize=(7, 5))
    for solver_string in ['ant', date1 + '_sim', date2 + '_sim']:
        with open(solver_string + '_pathlengths_all_states.json', 'r') as f:
            json_file = json.load(f)
        filenames = json_file.keys()
        translation_in_states, rotation_in_states = calc_trans_rot_per_state_per_filename(filenames, json_file)

        # sum up translation for states 'ac' and 'ab'
        trans_ac_ab = {filename: translation_in_states['ac'][filename] + translation_in_states['ab'][filename]
                       for filename in filenames if 'XL' in filename}

        df_trans_ac_ab = pd.DataFrame.from_dict({'filename': trans_ac_ab.keys(), 'trans_ab_ac': trans_ac_ab.values()}, )
        df_trans_ac_ab = df_trans_ac_ab.sort_values('trans_ab_ac').reset_index(drop=True)

        # choose the top, the middle, and the tail
        best, middle, worst = [(df_trans_ac_ab.iloc[i].filename, df_trans_ac_ab.iloc[i].trans_ab_ac)
                               for i in [0, len(df_trans_ac_ab) // 2, -1]]
        print('best: ' + str(best))
        print('middle: ' + str(middle))
        print('worst: ' + str(worst))

        # plot distribution of trans_ac_ab
        ax.hist(df_trans_ac_ab['trans_ab_ac'], bins=20, alpha=0.5, label=solver_string)
        ax.legend()
    DEBUG = 1


def plot_percentile_40th(percentile_40th, axs_40):
    percentile_40th['ant']['XS'] = np.nan
    percentile_40th['ant']['S'] = percentile_40th['ant']['S (> 1)']
    order = ['XS', 'S', 'M', 'L', 'XL']
    for solver_string, label, linestyle, color in zip(['ant', date + '_sim', date1 + '_sim'],
                                                      ['exp', 'sim_no_removal', 'sim_with_removal'],
                                                      ['-', '--', '--'],
                                                      ['k', 'blue', 'k']):
        y = [percentile_40th[solver_string][size] for size in order]
        axs_40[0].plot(order, y, label=label, linewidth=2, marker='o', markersize=10, color=color, linestyle=linestyle)
    axs_40[0].legend()
    axs_40[0].set_xlabel('size')
    axs_40[0].set_title('ants')

    order = ['Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']
    y = [percentile_40th['human'][size] for size in order]
    axs_40[1].plot(order, y, label='human', linewidth=2, marker='o', markersize=10, color='black', linestyle='-')
    axs_40[1].legend()
    axs_40[1].set_xlabel('size')
    axs_40[1].set_title('humans')


if __name__ == '__main__':
    # finding_extremes()
    percentile_40th_rot = {}
    percentile_40th_trans = {}

    for solver_string in tqdm(['ant', date1 + '_sim', 'human', date + '_sim',], desc='solver'):
        with open(solver_string + '_pathlengths_all_states.json', 'r') as f:
            json_file = json.load(f)

        filenames = json_file.keys()
        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]
        scale_trans = scale_trans_per_solver[solver]
        scale_rot = scale_rot_per_solver[solver]
        df = {'ant': df_ant_excluded,
              'SimTrjs_RemoveAntsNearWall=True_sim': df_gillespie1,
              'SimTrjs_RemoveAntsNearWall=False_sim': df_gillespie,
              'human': df_human}[solver_string]

        translation_in_states, rotation_in_states = calc_trans_rot_per_state_per_filename(filenames, json_file)

        plot_trans_rot_in_histograms(df, translation_in_states, rotation_in_states)
        plt.close()

        calc_trans_rot_per_state_per_filename_measure_without_state()

        with open(solver_string + '_trans.json', 'r') as f:
            trans = json.load(f)

        with open(solver_string + '_rot.json', 'r') as f:
            rot = json.load(f)

        plot_CDF(df, trans, scale_trans, xlim=xlim_per_solver_trans[solver], minimal=minimal_scaled_translation)
        plt.xlabel('translation / maze width')
        plt.savefig('results\\' + solver_string + '_CFDs_trans.png')
        plt.savefig('results\\' + solver_string + '_CFDs_trans.svg')
        plt.close()

        plot_CDF(df, rot, scale_rot, xlim=xlim_per_solver_rot[solver], minimal=minimal_scaled_rotation)
        plt.xlabel('rotation / 2pi')
        plt.savefig('results\\' + solver_string + '_CFDs_rot.png')
        plt.savefig('results\\' + solver_string + '_CFDs_rot.svg')
        plt.close()

        percentile_40th_trans[solver_string] = calc_40th_percentile(df, trans, scale_trans, solver)
        percentile_40th_rot[solver_string] = calc_40th_percentile(df, rot, scale_rot, solver)

    fig_40, axs_40 = plt.subplots(1, 2, figsize=(10, 4))
    plot_percentile_40th(percentile_40th_trans, axs_40)
    axs_40[0].set_ylabel('trans/arena height (40th percentile)')
    axs_40[1].set_ylabel('trans/arena height (40th percentile)')
    plt.tight_layout()
    plt.savefig('results\\40th_percentile_translation.png')
    plt.savefig('results\\40th_percentile_translation.svg', transparent=True)

    fig_40, axs_40 = plt.subplots(1, 2, figsize=(10, 4))
    plot_percentile_40th(percentile_40th_rot, axs_40)
    axs_40[0].set_ylabel('rot/ 2 * pi (40th percentile)')
    axs_40[1].set_ylabel('rot/ 2 * pi (40th percentile)')
    plt.tight_layout()
    plt.savefig('results\\40th_percentile_rotation.png')
    plt.savefig('results\\40th_percentile_rotation.svg', transparent=True)
    #
    # trans_without_state_cg = \
    #     {filename: measure_without_state(filename, ['cg'], translation_in_states)
    #      for filename in filenames};
    #
    # with open(solver + '_trans_without_state_cg.json', 'w') as f:
    #     json.dump(trans_without_state_cg, f)
    #
    # rot_without_state_cg = \
    #     {filename: measure_without_state(filename, ['cg'], rotation_in_states)
    #      for filename in filenames}
    #
    # with open(solver + '_rot_without_state_cg.json', 'w') as f:
    #     json.dump(rot_without_state_cg, f)
    #
    # with open(solver + '_trans_without_state_cg.json', 'r') as f:
    #     trans_without_state_cg = json.load(f)
    #
    # with open(solver + '_rot_without_state_cg.json', 'r') as f:
    #     rot_without_state_cg = json.load(f)
    #
    # plot_CDF(trans_without_state_cg)
    # plt.savefig(solver + '_CFDs_trans_no_cg.png')
    # plt.close()
    #
    # plot_CDF(rot_without_state_cg)
    # plt.savefig(solver + '_CFDs_rot_no_cg.png')
    # plt.close()
