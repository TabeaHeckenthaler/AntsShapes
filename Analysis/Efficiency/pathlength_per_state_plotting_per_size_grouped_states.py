from pathlength_per_state_plotting_per_size import *


linestyle_solver = {'ant': '-', 'sim': '--', 'human': '-'}
ylim = {'ant': 25, 'sim': 25, 'human': 1}

# df_gillespie delete 'XS' from dataframe
df_gillespie1 = df_gillespie1[df_gillespie1['size'] != 'XS']

super_states = {'a': ['ab', 'ac', 'b'],
                'b': ['b1', 'b2'],
                'c': ['c', 'cg'],
                'e': ['e', 'eg']
                }

color_states = {'a': 'red', 'b': '#8c34eb', 'c': '#eb9b34', 'e': 'grey'}


def plot_histograms(df, translation_in_states, ax, scale):
    groups = get_size_groups(df, solver)

    trans_mean = {}
    trans_std = {}
    num_trjs = {}

    for size, group in groups.items():
        filenames = group['filename'].tolist()
        trans_mean[size] = {}
        trans_std[size] = {}
        for super_state, states in super_states.items():
            trans_mean[size][super_state] = 0
            trans_std[size][super_state] = 0
            for state in states:
                trans_mean[size][super_state] += np.mean([translation_in_states[state][filename]
                                                          for filename in filenames])
                trans_std[size][super_state] += np.std([translation_in_states[state][filename]
                                                 for filename in filenames]) / np.sqrt(len(filenames))
        num_trjs[size] = len(filenames)

    # if trans_mean contains key S (> 1), then rename it to S
    if 'S (> 1)' in trans_mean.keys():
        trans_mean['S'] = trans_mean.pop('S (> 1)')
        trans_std['S'] = trans_std.pop('S (> 1)')
    sizes = ['S', 'M', 'L', 'XL']

    for super_state in super_states.keys():
        if super_state == 'a':
            DEBUG =1
        # it would be best to plot all into the same histogram
        ax.errorbar(sizes,
                    [trans_mean[size][super_state] / scale[size] for size in sizes],
                    yerr=[trans_std[size][super_state] / scale[size] for size in sizes],
                    label=super_state + '_' + solver,
                    color=color_states[super_state],
                    linewidth=2,
                    marker='*',
                    linestyle=linestyle_solver[solver])


# def plot_CDF(df, measure, scale, ax):
#     df['measure'] = df['filename'].map(measure)
#     dfs = get_size_groups(df, solver, only_winner=False)
#
#     for size, df_size in tqdm(dfs.items()):
#         df_individual = df[df['filename'].isin(df_size['filename'])]
#
#         df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(scale)
#
#         # sort by norm measure
#         df_individual = df_individual.sort_values(by='norm measure')
#         longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
#         df_individual = df_individual[(df_individual['norm measure'] > longest_succ_experiment)
#                                       | (df_individual['winner'])]
#         x_values = np.arange(0, df_individual['norm measure'].max() + 2 * solver_step[solver], step=solver_step[solver])
#         y_values = []
#
#         for x in x_values:
#             suc = df_individual[(df_individual['norm measure'] < x) & (df_individual['winner'])]
#             y_values.append(len(suc) / len(df_individual))
#         ax.step(x_values, y_values,
#                 label=size + '_' + solver + ' : ' + str(len(df_individual)),
#                 color=colors[size],
#                 linewidth=2,
#                 where='post',
#                 linestyle=linestyle_solver[solver])


if __name__ == '__main__':

    fig_hist, axs_hist = plt.subplots(1, 2, figsize=(2 * 4, 6))
    # fig_CDF, axs_CDF = plt.subplots(1, 2, figsize=(2 * 4, 3))

    with open('ant_trans.json', 'r') as f:
        trans_ant = json.load(f)

    with open('ant_rot.json', 'r') as f:
        rot_ant = json.load(f)

    with open(date1 + '_sim_trans.json', 'r') as f:
        trans_sim = json.load(f)

    with open(date1 + '_sim_rot.json', 'r') as f:
        rot_sim = json.load(f)

    ant_trans_fs, sim_trans_fs = filenames_to_include(df_ant_excluded, df_gillespie1, trans_ant, trans_sim,
                                                      scale_trans_per_solver, perc_all=0.45)
    ant_rot_fs, sim_rot_fs = filenames_to_include(df_ant_excluded, df_gillespie1, rot_ant, rot_sim,
                                                  scale_rot_per_solver, perc_all=0.45)

    for solver_string in ['ant', date1 + '_sim']:
        with open(solver_string + '_pathlengths_all_states.json', 'r') as f:
            json_file = json.load(f)
        filenames = json_file.keys()
        translation_in_states, rotation_in_states = calc_trans_rot_per_state_per_filename(filenames, json_file)

        solver = solver_string.split('_')[-1]
        sizes = sizes_per_solver[solver]

        scale_trans = scale_trans_per_solver[solver]
        scale_rot = scale_rot_per_solver[solver]
        df = {'ant': df_ant_excluded, 'sim': df_gillespie1, 'human': df_human}[solver]

        df_trans = df.copy()
        df_trans = df_trans[(df_trans['filename'].isin(ant_trans_fs + sim_trans_fs))]

        plot_histograms(df_trans, translation_in_states, axs_hist[0], scale_trans)
        # with open(solver_string + '_trans.json', 'r') as f:
        #     trans = json.load(f)
        # plot_CDF(df, trans, scale_trans, axs_CDF[0])

        df_rot = df.copy()
        df_rot = df_rot[df_rot['filename'].isin(ant_rot_fs + sim_rot_fs)]

        plot_histograms(df_rot, rotation_in_states, axs_hist[1], scale_rot)
        # with open(solver_string + '_rot.json', 'r') as f:
        #     rot = json.load(f)
        # plot_CDF(df, rot, scale_rot, axs_CDF[1])

    axs_hist[1].legend()
    axs_hist[0].set_ylabel('translation / maze width')
    axs_hist[1].set_ylabel('rotation / 2 * pi')
    if solver in ['ant', date1 + '_sim']:
        axs_hist[0].set_ylim([0, 25])
        axs_hist[1].set_ylim([0, 8])
    fig_hist.suptitle(solver_string)
    fig_hist.tight_layout()
    fig_hist.savefig('results\\' + solver_string + '_per_super_state.png')
    fig_hist.savefig('results\\' + solver_string + '_per_super_state.svg', transparent=True)
    fig_hist.savefig('results\\' + solver_string + '_per_super_state.pdf', transparent=True)

    # if solver in ['ant', date + '_sim']:
    #     axs_CDF[0].set_xlim([0, 100])
    #     axs_CDF[1].set_xlim([0, 60])
    # axs_CDF[0].set_xlabel('translation / maze width')
    # axs_CDF[1].set_xlabel('rotation / 2 * pi')
    # axs_CDF[0].axvline(minimal_scaled_translation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    # axs_CDF[0].text(minimal_scaled_translation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    # axs_CDF[1].axvline(minimal_scaled_rotation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    # axs_CDF[1].text(minimal_scaled_rotation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    # axs_CDF[0].set_ylabel('% success')
    # axs_CDF[0].set_ylim([0, 1])
    # axs_CDF[1].set_ylim([0, 1])
    # axs_CDF[0].legend(prop={'size': 12})
    # # fig_CDF.suptitle(solver_string)
    # fig_CDF.tight_layout()
    # fig_CDF.savefig('results\\' + solver_string + '_CFDs.png')
    # fig_CDF.savefig('results\\' + solver_string + '_CFDs.svg', transparent=True)
