from pathlength_per_state_plotting import *

linestyle_solver = {'ant': '-', 'sim': 'dotted', 'human': '-'}
marker_solver = {'ant': '*', 'sim': '.', 'human': '*'}
ylim = {'ant': 25, 'sim': 25, 'human': 1}

# df_gillespie delete 'XS' from dataframe
df_gillespie = df_gillespie[df_gillespie['size'] != 'XS']


def filenames_to_include(df_ant, df_sim, measure_ant, measure_sim, scale, perc_all=None):
    df_sim['measure'] = df_sim['filename'].map(measure_sim)

    df_ant['measure'] = df_ant['filename'].map(measure_ant)
    dfs_ant = get_size_groups(df_ant, 'ant')

    percentile = {}
    filenames_to_include_ant = []
    for size, df_size in tqdm(dfs_ant.items()):
        df_individual = df_ant_excluded[df_ant_excluded['filename'].isin(df_size['filename'])]

        df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(scale['ant'])

        # sort by norm measure
        df_individual = df_individual.sort_values(by='norm measure')
        longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
        df_individual = df_individual[(df_individual['norm measure'] > longest_succ_experiment)
                                      | (df_individual['winner'])]

        # find percentage of 'winner'
        if perc_all is None:
            percentile[size] = len(df_individual[df_individual['winner']]) / len(df_individual)
            filenames_to_include_ant.extend(df_individual['filename'].tolist())
        else:
            percentile[size] = perc_all
            df_individual = df_individual[df_individual['norm measure'] < np.quantile(df_individual['norm measure'],
                                                                                      percentile[size])]
            filenames_to_include_ant.extend(df_individual['filename'].tolist())

    dfs_gillespie = get_size_groups(df_sim, 'sim')
    filenames_to_include_sim = []
    percentile['S'] = percentile['S (> 1)']
    percentile['XS'] = 1

    for size, df_size in tqdm(dfs_gillespie.items()):
        df_individual = df_sim[df_sim['filename'].isin(df_size['filename'])]
        df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(scale['sim'])

        # for df_sim cut out the experiments that are not part of the percentile[size]
        df_individual = df_individual.sort_values(by='norm measure')
        # reindex
        df_individual = df_individual.reset_index(drop=True)
        df_individual = df_individual[df_individual['norm measure'] < np.quantile(df_individual['norm measure'],
                                                                                  percentile[size])]
        DEBUG = 1
        filenames_to_include_sim.extend(df_individual['filename'].tolist())
    return filenames_to_include_ant, filenames_to_include_sim


def plot_histograms(df, filenames_to_include, translation_in_states, ax, scale, solver):
    groups = get_size_groups(df, solver)

    trans_mean = {}
    trans_std = {}
    num_trjs = {}

    for size, group in groups.items():
        filenames = set(group['filename'].tolist())
        filenames = filenames.intersection(set(filenames_to_include))
        trans_mean[size] = {}
        trans_std[size] = {}
        for state in states:
            trans_mean[size][state] = np.mean([translation_in_states[state][filename] for filename in filenames])
            trans_std[size][state] = np.std([translation_in_states[state][filename]
                                             for filename in filenames]) / np.sqrt(len(filenames))
        num_trjs[size] = len(filenames)

    for size in sizes_per_solver[solver]:
        # it would be best to plot all into the same histogram
        ax.errorbar(states, [trans_mean[size][state] / scale[size] for state in states],
                    yerr=[trans_std[size][state] / scale[size] for state in states],
                    label=size + '_' + solver + ' : ' + str(num_trjs[size]),
                    color=colors[size],
                    linewidth=2,
                    marker=marker_solver[solver],
                    linestyle=linestyle_solver[solver])


def plot_CDF(df, measure, scale, ax, solver):
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
        x_values = np.arange(0, df_individual['norm measure'].max() + 2 * solver_step[solver], step=solver_step[solver])
        y_values = []

        for x in x_values:
            suc = df_individual[(df_individual['norm measure'] < x) & (df_individual['winner'])]
            y_values.append(len(suc) / len(df_individual))
        ax.step(x_values, y_values,
                label=size + '_' + solver + ' : ' + str(len(df_individual)),
                color=colors[size],
                linewidth=2,
                where='post',
                linestyle=linestyle_solver[solver])


def plot_CDFs_exp_sim(sim_condition, df_sim):
    fig_CDF, axs_CDF = plt.subplots(1, 2, figsize=(4 * 4, 6))

    for solver_string in ['ant', sim_condition + '_sim']:
        solver = solver_string.split('_')[-1]
        scale_trans = scale_trans_per_solver[solver]
        scale_rot = scale_rot_per_solver[solver]
        df = {'ant': df_ant_excluded, 'sim': df_sim, 'human': df_human}[solver]

        with open(solver_string + '_trans.json', 'r') as f:
            trans = json.load(f)
        plot_CDF(df, trans, scale_trans, axs_CDF[0], solver_string.split('_')[-1])

        with open(solver_string + '_rot.json', 'r') as f:
            rot = json.load(f)
        plot_CDF(df, rot, scale_rot, axs_CDF[1], solver_string.split('_')[-1])
        if solver in ['ant', sim_condition + '_sim']:
            axs_CDF[0].set_xlim([0, 350])
            axs_CDF[1].set_xlim([0, 100])
    axs_CDF[0].set_xlabel('translation / maze width')
    axs_CDF[1].set_xlabel('rotation / 2 * pi')
    axs_CDF[0].axvline(minimal_scaled_translation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    axs_CDF[0].text(minimal_scaled_translation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    axs_CDF[1].axvline(minimal_scaled_rotation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    axs_CDF[1].text(minimal_scaled_rotation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    axs_CDF[0].set_ylabel('% success')
    axs_CDF[0].set_ylim([0, 1])
    axs_CDF[1].set_ylim([0, 1])
    axs_CDF[0].legend(prop={'size': 12})

    fig_CDF.tight_layout()
    fig_CDF.savefig('results\\' + 'ant_exp_sim_CFDs.png')
    fig_CDF.savefig('results\\' + 'ant_exp_sim_CFDs.svg', transparent=True)
    fig_CDF.savefig('results\\' + 'ant_exp_sim_CFDs.pdf', transparent=True)


def plot_CDFs_human():
    fig_CDF, axs_CDF = plt.subplots(1, 2, figsize=(4 * 4, 6))

    solver_string = 'human'
    solver = solver_string
    scale_trans = scale_trans_per_solver[solver]
    scale_rot = scale_rot_per_solver[solver]
    df = df_human

    with open(solver_string + '_trans.json', 'r') as f:
        trans = json.load(f)
    plot_CDF(df, trans, scale_trans, axs_CDF[0], solver_string.split('_')[-1])

    with open(solver_string + '_rot.json', 'r') as f:
        rot = json.load(f)
    plot_CDF(df, rot, scale_rot, axs_CDF[1], solver_string.split('_')[-1])
    # axs_CDF[0].set_xlim([0, 100])
    # axs_CDF[1].set_xlim([0, 60])
    axs_CDF[0].set_xlabel('translation / maze width')
    axs_CDF[1].set_xlabel('rotation / 2 * pi')
    axs_CDF[0].axvline(minimal_scaled_translation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    axs_CDF[0].text(minimal_scaled_translation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    axs_CDF[1].axvline(minimal_scaled_rotation, color='black', linestyle='--', linewidth=2, alpha=0.5)
    axs_CDF[1].text(minimal_scaled_rotation + 0.1, 0.5, 'minimal', rotation=90, fontsize=20, alpha=0.5)
    axs_CDF[0].set_ylabel('% success')
    axs_CDF[0].set_ylim([0, 1])
    axs_CDF[1].set_ylim([0, 1])
    axs_CDF[0].legend(prop={'size': 12})

    fig_CDF.tight_layout()
    fig_CDF.savefig('results\\' + 'human_CFDs.png')
    fig_CDF.savefig('results\\' + 'human_CFDs.svg', transparent=True)
    fig_CDF.savefig('results\\' + 'human_CFDs.pdf', transparent=True)


def plot_histograms_ant_exp_sim(sim_condition, df_sim):
    fig_hist, axs_hist = plt.subplots(1, 2, figsize=(4 * 4, 6))

    with open('ant_trans.json', 'r') as f:
        trans_ant = json.load(f)

    with open('ant_rot.json', 'r') as f:
        rot_ant = json.load(f)

    with open(sim_condition + '_sim_trans.json', 'r') as f:
        trans_sim = json.load(f)

    with open(sim_condition + '_sim_rot.json', 'r') as f:
        rot_sim = json.load(f)

    with open(sim_condition + '_sim_pathlengths_all_states.json', 'r') as f:
        json_file = json.load(f)
    filenames = json_file.keys()
    translation_in_states_sim, rotation_in_states_sim = calc_trans_rot_per_state_per_filename(filenames, json_file)

    with open('ant_pathlengths_all_states.json', 'r') as f:
        json_file = json.load(f)
    filenames = json_file.keys()
    translation_in_states_ant, rotation_in_states_ant = calc_trans_rot_per_state_per_filename(filenames, json_file)

    ant_trans_fs, sim_trans_fs = filenames_to_include(df_ant_excluded, df_sim, trans_ant, trans_sim,
                                                      scale_trans_per_solver, perc_all=0.45)
    ant_rot_fs, sim_rot_fs = filenames_to_include(df_ant_excluded, df_sim,
                                                  rot_ant, rot_sim, scale_rot_per_solver, perc_all=0.45)



    plot_histograms(df_ant_excluded, ant_trans_fs, translation_in_states_ant, axs_hist[0],
                    scale_trans_per_solver['ant'], 'ant')
    plot_histograms(df_ant_excluded, ant_rot_fs, rotation_in_states_ant, axs_hist[1], scale_rot_per_solver['ant'],
                    'ant')

    plot_histograms(df_sim, sim_trans_fs, translation_in_states_sim, axs_hist[0], scale_trans_per_solver['sim'], 'sim')
    plot_histograms(df_sim, sim_rot_fs, rotation_in_states_sim, axs_hist[1], scale_rot_per_solver['sim'], 'sim')

    axs_hist[0].legend()
    axs_hist[0].set_ylabel('translation / maze width')
    axs_hist[1].set_ylabel('rotation / 2 * pi')
    axs_hist[0].set_ylim([0, 12])
    axs_hist[1].set_ylim([0, 5])
    fig_hist.tight_layout()
    fig_hist.savefig('results\\' + 'pL_per_state_40th_percentile.png')
    fig_hist.savefig('results\\' + 'pL_per_state_40th_percentile.pdf', transparent=True)
    fig_hist.savefig('results\\' + 'pL_per_state_40th_percentile.svg', transparent=True)


def plot_histograms_human(df_human):
    fig_hist, axs_hist = plt.subplots(1, 2, figsize=(4 * 4, 6))

    with open('human_pathlengths_all_states.json', 'r') as f:
        json_file = json.load(f)
    filenames = json_file.keys()

    translation_in_states, rotation_in_states = calc_trans_rot_per_state_per_filename(filenames, json_file)

    plot_histograms(df_human, df_human['filename'], translation_in_states, axs_hist[0], scale_trans_per_solver['human'],
                    'human')
    plot_histograms(df_human, df_human['filename'], rotation_in_states, axs_hist[1], scale_rot_per_solver['human'],
                    'human')

    axs_hist[0].legend()
    axs_hist[0].set_ylabel('translation / maze width')
    axs_hist[1].set_ylabel('rotation / 2 * pi')
    axs_hist[0].set_ylim([0, 1.2])
    axs_hist[1].set_ylim([0, 0.4])
    fig_hist.tight_layout()
    fig_hist.savefig('results\\' + 'pL_per_state_human.png')
    fig_hist.savefig('results\\' + 'pL_per_state_human.pdf', transparent=True)
    fig_hist.savefig('results\\' + 'pL_per_state_human.svg', transparent=True)


def plot_percentile_40th(percentile_40th, axs_40):
    percentile_40th['ant']['XS'] = np.nan
    percentile_40th['ant']['S'] = percentile_40th['ant']['S (> 1)']
    order = ['XS', 'S', 'M', 'L', 'XL']
    for solver_string, label, linestyle, color in zip(['ant', date1 + '_sim'],
                                                      ['exp', 'sim_with_removal'],
                                                      ['-', '--'],
                                                      ['k', 'green']):
        y = [percentile_40th[solver_string][size] for size in order]
        axs_40.plot(order, y, label=label, linewidth=2, marker='*', markersize=10, color=color, linestyle=linestyle)
    axs_40.legend()
    axs_40.set_xlabel('size')

    # order = ['Large C', 'Large NC', 'Medium C', 'Medium NC', 'Small']
    # y = [percentile_40th['human'][size] for size in order]
    # axs_40[1].plot(order, y, label='human', linewidth=2, marker='o', markersize=10, color='black', linestyle='-')
    # axs_40[1].legend()
    # axs_40[1].set_xlabel('size')
    # axs_40[1].set_title('humans')


def plot_40th_ant_human():
    percentile_40th_rot = {}
    percentile_40th_trans = {}

    for solver_string in tqdm([date1 + '_sim', 'ant'], desc='solver'):
        with open(solver_string + '_trans.json', 'r') as f:
            trans = json.load(f)

        with open(solver_string + '_rot.json', 'r') as f:
            rot = json.load(f)

        df = {'ant': df_ant_excluded,
              'SimTrjs_RemoveAntsNearWall=True_sim': df_gillespie1,
              'SimTrjs_RemoveAntsNearWall=False_sim': df_gillespie,
              'human': df_human}[solver_string]
        solver = solver_string.split('_')[-1]
        percentile_40th_trans[solver_string] = calc_40th_percentile(df, trans, scale_trans_per_solver[solver], solver)
        percentile_40th_rot[solver_string] = calc_40th_percentile(df, rot, scale_rot_per_solver[solver], solver)

    fig_40, axs_40 = plt.subplots(1, 2, figsize=(10, 4))
    plot_percentile_40th(percentile_40th_trans, axs_40[0])
    axs_40[0].set_ylabel('trans/arena height (40th perc)')
    axs_40[0].set_ylabel('trans/arena height (40th perc)')

    plot_percentile_40th(percentile_40th_rot, axs_40[1])
    axs_40[1].set_ylabel('rot/ 2 * pi (40th perc)')
    axs_40[1].set_ylabel('rot/ 2 * pi (40th perc)')
    plt.tight_layout()
    plt.savefig('results\\40th_percentile.png')
    plt.savefig('results\\40th_percentile.svg', transparent=True)
    plt.savefig('results\\40th_percentile.pdf', transparent=True)


if __name__ == '__main__':
    # plot_CDFs_human()
    # plot_CDFs_exp_sim(date1, df_gillespie1)
    # plot_histograms_ant_exp_sim(date1, df_gillespie1)
    plot_histograms_human(df_human)
    plot_40th_ant_human()

    DEBUG = 1
