from StateMachine import *
from DataFrame.import_excel_dfs import dfs_human
from Directories import network_dir

with open(os.path.join(network_dir, 'state_series_selected_states.json'), 'r') as json_file:
    state_series_dict = json.load(json_file)
    json_file.close()

df_exp = dfs_human['Small']
df_exp['state series'] = df_exp['filename'].map(state_series_dict)


def find_means(df, normalize=True):
    # find the percentage of every state in the state series
    states = ['ab', 'ac', 'b', 'be', 'b1', 'b2', 'b1/b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h']
    state_percentages = pd.DataFrame(0, index=df['filename'], columns=states)
    for filename, ts in zip(df['filename'], df['state series']):
        state_percentages.loc[filename] = pd.Series(ts).value_counts(normalize=normalize)

    state_percentages['b1/b2'] = (state_percentages['b1'] + state_percentages['b2'])/2
    # change position column 'b1/b2' to come after 'ac'
    state_percentages.drop(columns=['b1', 'b2'], inplace=True)

    state_percentages.fillna(0, inplace=True)

    # find the average percentage of every state
    state_percentages_mean = state_percentages.mean()
    return state_percentages_mean


def plot_state_percentages(state_percentages_mean, ax, **kwargs):
    # plot state_percentages_mean empty bars
    ax.bar(state_percentages_mean.index, state_percentages_mean.values,
           color=[colors_state[state] for state in state_percentages_mean.index],
           edgecolor=[colors_state[state] for state in state_percentages_mean.index], **kwargs)


def loss(p1: np.array, p2: np.array) -> float:
    return np.sqrt(np.sum((p1.to_numpy() - p2.to_numpy()) ** 2))


def compare(name, plot=False) -> float:
    df_sim = HumanStateMachine.df(name=name + "_states_small.json")
    df_sim.rename(columns={'name': 'filename', 'states_series': 'state series'}, inplace=True)

    state_percentages_mean_exp = find_means(df_exp, normalize=False)
    state_percentages_mean_sim = find_means(df_sim, normalize=False)

    if plot:
        fig_percentages, ax_percentages = plt.subplots()
        plot_state_percentages(state_percentages_mean_exp, ax_percentages, fill=False, linewidth=4, label='experiment')
        plot_state_percentages(state_percentages_mean_sim, ax_percentages, alpha=0.3, label='simulation')
        plt.legend()
        save_fig(fig_percentages, name='times_entered_' + name)

    loss_value = loss(state_percentages_mean_exp, state_percentages_mean_sim)
    plt.close('all')
    return loss_value


if __name__ == '__main__':
    resolution = 10
    pattern_recognitions = np.linspace(0.1, 1, resolution)
    # biass = np.linspace(1.05, 1.2, resolution)
    biass = [1.1]
    # weakening_factors = np.linspace(0.00001, 0.1, resolution)
    # weakening_factors = [0.00001]
    weakening_factors = [0]

    # with open('loss_values.json', 'r') as json_file:
    #     loss_values = np.array(json.load(json_file))
    #     json_file.close()
    #
    # # find indices with maximal values in loss_values
    # indices = np.stack(np.where(loss_values == np.nanmin(loss_values))).squeeze()
    # best_values = [pattern_recognitions[indices[0]], biass[indices[1]], weakening_factors[indices[2]]]
    # print('best values: ' + str(best_values))

    DEBUG = 1

    state_percentages_mean_exp = find_means(df_exp, normalize=False)

    # open a three-dimensional numpy array to save the loss values
    # the dimensions represent 'pattern_recognition', 'bias' and 'weakening_factor'
    loss_values = np.zeros((len(pattern_recognitions), len(biass), len(weakening_factors)))
    loss_values.fill(np.nan)
    # save loss_values json file
    with open('loss_values.json', 'w') as json_file:
        json.dump(loss_values.tolist(), json_file)
        json_file.close()

    for (i1, i2, i3), _ in tqdm(np.ndenumerate(loss_values), total=loss_values.size):

        pattern_recognition, bias, weakening_factor = pattern_recognitions[i1], biass[i2], weakening_factors[i3]
        paths = []
        for i in range(20):
            stateMachine = HumanStateMachine(seed=i,
                                             pattern_recognition=pattern_recognition,
                                             bias=bias,
                                             weakening_factor=weakening_factor)
            stateMachine.run(cutoff=100)
            paths.append(stateMachine.path)

        name = 'pattern_recognition_' + str(pattern_recognition) + '_ineg_bias_' + str(bias) + \
               '_weakening_factor_' + str(weakening_factor)
        HumanStateMachine.save(paths, name)

        loss_value = compare(name, plot=True)
        print("pattern_recognition: " + str(pattern_recognition) + " bias: " + str(bias) + " weakening_factor: " + str(
            weakening_factor))
        print('loss: ' + str(loss_value))

        # save loss_values json file
        with open('loss_values.json', 'r') as json_file:
            loss_values = np.array(json.load(json_file))
            json_file.close()

        loss_values[i1, i2, i3] = loss_value

        with open('loss_values.json', 'w') as json_file:
            json.dump(loss_values.tolist(), json_file)
            json_file.close()

    with open('loss_values.json', 'r') as json_file:
        loss_values = np.array(json.load(json_file))
        json_file.close()

    # find indices with maximal values in loss_values
    indices = np.stack(np.where(loss_values == np.nanmin(loss_values))).squeeze()
    best_values = [pattern_recognitions[indices[0]], biass[indices[1]], weakening_factors[indices[2]]]
    print('best values: ' + str(best_values))
