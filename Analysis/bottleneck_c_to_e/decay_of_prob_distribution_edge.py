import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from Analysis.bottleneck_c_to_e.correlation_edge_walk_decision_c_e_ac import *
from Setup.Maze import Maze
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates
from skfmm import distance
import matplotlib.lines as lines
import cv2
from PhysicsEngine.Display import Display
import pygame


def plot_traj_in_2D(traj, bool_to_plot=None, ax=None):
    """
    Plot the trajectories in the config space.
    """

    # plot the x and y coordinates of self.traj
    if bool_to_plot is None:
        bool_to_plot = np.ones_like(traj.angle).astype(bool)

    if ax is None:
        fig, ax = plt.subplots()

    for bool_to_plot_, color in zip([bool_to_plot, np.logical_not(bool_to_plot)],
                                    [(0.96298491, 0.6126247, 0.45145074),
                                     (0.01060815, 0.01060815, 0.01060815)]):
        # plot the traj.position[bool_to_plot_, 0] and traj.position[bool_to_plot_, 1] in fig
        ax.scatter(traj.position[bool_to_plot_, 0], traj.position[bool_to_plot_, 1], color=color, lw=0.5)
    plt.axis('equal')


def title_for_saving(title: str) -> str:
    # replace in the title the characters that are not allowed in a filename
    title = title.replace(' ', '_')
    title = title.replace('(', '')
    title = title.replace(')', '')
    title = title.replace('[', '')
    title = title.replace(']', '')
    title = title.replace('<', '_less_than_')
    title = title.replace('>', '_more_than_')
    return title


def plot_2d_density(x, y, maze=None, title=''):
    # make a heatmap of x and y with plt
    if maze is None:
        f_title = {'XL': 1, 'L': 1 / 2, 'M': 1 / 4, 'S (> 1)': 1 / 8, 'Single (1)': 1 / 8}
        f = f_title[title]
        r = [[13 * f, 19 * f], [0 * f, 19.1 * f]]
    else:
        r = [[maze.slits[0], maze.slits[1]], [0, maze.arena_height]]

    h, x_edges, y_edges, image = plt.hist2d(x, y, bins=100, cmap='viridis', range=r)
    # h = np.log(h)

    ax = plt.imshow(h)
    ax.axes.set_title(title)
    # equal aspect ratio
    ax.axes.set_aspect('equal')

    # replace in the title the characters that are not allowed in a filename
    title = title_for_saving(title)

    # save the figure without the white border with high resolution
    plt.gcf().savefig('2d_density_log' + title + '.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def save_coords():
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # for size, df in [(size, df) for size, df in dfs_ant.items() if size not in ['Single (1)']]:
    for size, df in zip(['S'], [dfs_ant['S (> 1)']]):
        all_coords_x = {}
        all_coords_y = {}
        all_coords_theta = {}
        all_coords_frames = {}
        # ps = ConfigSpace_SelectedStates(solver='ant', size=size, shape='SPT',
        #                                 geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
        #                                           'LoadDimensions_new2021_SPT_ant.xlsx'))
        # ps.load_final_labeled_space()
        # c_space = np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg')

        # add title to tqdm

        for filename in tqdm(df['filename'], desc=size):
            # maze = Maze(get(df['filename'].iloc[0]))

            traj = get(filename)

            if traj.solver == 'ant' and \
                    traj.geometry() != ('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'):
                traj = traj.confine_to_new_dimensions()
            traj.smooth(1)

            ts = time_series_dict[filename]
            in_c_trajs, out_c_trajs = In_the_bottle.cut_traj(traj, ts, buffer=2)

            for in_c_traj in in_c_trajs:
                # inside = np.array([ps.space[ps.coords_to_indices(x, y, theta)]
                #                    for x, y, theta in
                #                    zip(in_c_traj.position[:, 0], in_c_traj.position[:, 1], in_c_traj.angle)])
                # print(np.sum(inside) / len(inside))

                # while np.sum(inside) / len(inside) < 0.75:
                #     correct_trajectory(in_c_traj, c_space, inside, ps)
                #     in_c_traj.position[:, 0] = in_c_traj.position[:, 0] - float(input())
                #     inside = np.array([ps.space[ps.coords_to_indices(x, y, theta)]
                #                        for x, y, theta in
                #                        zip(in_c_traj.position[:, 0], in_c_traj.position[:, 1], in_c_traj.angle)])

                coord_x = in_c_traj.position[:, 0][::in_c_traj.fps]
                coord_y = in_c_traj.position[:, 1][::in_c_traj.fps]
                coord_theta = in_c_traj.angle[::in_c_traj.fps]
                coords_frames = in_c_traj.frames[::in_c_traj.fps]

                all_coords_x.update({in_c_traj.filename + ': ' + str(coords_frames[0]): coord_x.tolist()})
                all_coords_y.update({in_c_traj.filename + ': ' + str(coords_frames[0]): coord_y.tolist()})
                all_coords_theta.update({in_c_traj.filename + ': ' + str(coords_frames[0]): coord_theta.tolist()})
                all_coords_frames.update({in_c_traj.filename + ': ' + str(coords_frames[0]): coords_frames.tolist()})

        with open(folder + size + 'all_coords_x.json', 'w') as json_file:
            json.dump(all_coords_x, json_file)
            json_file.close()

        with open(folder + size + 'all_coords_y.json', 'w') as json_file:
            json.dump(all_coords_y, json_file)
            json_file.close()

        with open(folder + size + 'all_coords_theta.json', 'w') as json_file:
            json.dump(all_coords_theta, json_file)
            json_file.close()

        with open(folder + size + 'all_coords_frames.json', 'w') as json_file:
            json.dump(all_coords_frames, json_file)
            json_file.close()


def find_largest_x(x, y, theta, ps, c_or_cg):
    ind = ps.coords_to_indices(0, y, theta)
    available_x_ind = np.where(c_or_cg[:, ind[1], ind[2]])[0]
    if len(available_x_ind) == 0:
        return x
    max_x_ind = max(available_x_ind)
    max_x = ps.indices_to_coords(max_x_ind, ind[1], ind[2])[0]
    return max_x


def power_law(x, a, b):
    return a * x ** b


def exp(x, a, b):
    return a * np.exp(-b * x)


def fit_exponential(x, y, func):
    popt, pcov = curve_fit(func, x, y)
    return popt, pcov


def fit_function_to_histogram(values: list, bins: list, func: callable, name: str):
    # find the middle between all the values
    x_bins = bins[:-1] + np.diff(bins) / 2

    popt, pcov = fit_exponential(x_bins, values, func=func)
    # popt, pcov = fit_power_law(x_bins, values)
    fit_results = pd.read_excel(folder + 'fit_results_' + name + '.xlsx', index_col=0)
    fit_results = fit_results.append(pd.DataFrame({'a': popt[0], 'b': popt[1],
                                                   'a_err': np.sqrt(pcov[0, 0]), 'b_err': np.sqrt(pcov[1, 1])},
                                                  index=[size]))

    fit_results.to_excel(folder + 'fit_results_' + name + '.xlsx')
    plt.bar(x_bins, values, width=np.diff(bins), align='edge', label='normalized calc_distances')
    plt.plot(x_bins, func(x_bins, popt[0], popt[1]), 'r--',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.legend()
    plt.title('Distance in x from boundary: ' + str(size))
    plt.savefig(folder + 'histogram_' + name + '_' + str(size) + '.png')
    plt.close()


def find_distances(size, all_coords_x, all_coords_y, all_coords_theta, all_coords_frames, shift=True, condition='_x'):
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    # plot_2d_density(np.concatenatSe(all_coords_x), np.concatenate(all_coords_y), maze=maze, title=str(size))

    # write a function that returns for every set of y and theta a number in x,
    # which is the largest permissible x.
    ps = ConfigSpace_SelectedStates(solver=maze.solver, size=maze.size, shape=maze.shape,
                                    geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                              'LoadDimensions_new2021_SPT_ant.xlsx'))
    ps.load_final_labeled_space()
    c_or_cg = np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg')

    # for every node in c_or_cg, find the distance to the boundary using the fast marching method
    if condition == '':
        dx = ps.indices_to_coords(1, 0, 0)[0]
        distances_from_boundary = distance(c_or_cg, dx=[dx for _ in range(3)])
        neg_distances_from_boundary = distance(~c_or_cg, dx=[dx for _ in range(3)])
        distances_from_boundary[~c_or_cg] = -neg_distances_from_boundary[~c_or_cg]

    elif condition == '_x':
        norm = maze.slits[1] - maze.slits[0]
        from PhysicsEngine.Display import Display
        import pygame
        d = Display('', 1, maze)
        percentages = {}

    all_distances = {}
    for i, key in tqdm(enumerate(all_coords_x.keys()), desc=size, total=len(all_coords_x.keys())):
        xs = all_coords_x[key]
        ys = all_coords_y[key]
        thetas = all_coords_theta[key]
        frames = all_coords_frames[key]

        if len(frames) > 60:
            if condition == '':
                distance_from_boundary = [distances_from_boundary[ps.coords_to_indices(x, y, theta)] for x, y, theta in
                                          zip(xs, ys, thetas)]

                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                values, bins, bar_container = axs[0].hist(distance_from_boundary, bins=100, density=True,
                                                          label='original')
                axs[1].plot(distance_from_boundary, label='original')

                if shift:
                    # raise NotImplementedError
                    max_bin = bins[np.argmax(values)]
                    distance_from_boundary -= max_bin
                    values, bins, bar_container = axs[0].hist(distance_from_boundary, bins=100, density=True,
                                                              label='shifted')
                    axs[1].plot(distance_from_boundary, label='shifted')
                    DEBUG = 1

                plt.legend()
                plt.savefig(
                    folder + 'single_exp_shift' + condition + '\\' + size + '_' + key.replace(': ',
                                                                                              '_') + '_histogram.png')
                plt.close()

                all_distances[key] = distance_from_boundary

                values, bins, bar_container = plt.hist(np.hstack(all_distances.values()), bins=30, density=True)
                # save values and bins
                with open(folder + size + 'values' + condition + '.json', 'w') as json_file:
                    json.dump(values.tolist(), json_file)
                    json_file.close()

                with open(folder + size + 'bins' + condition + '.json', 'w') as json_file:
                    json.dump(bins.tolist(), json_file)
                    json_file.close()

            elif condition == '_x':
                x_max = [find_largest_x(x, y, theta, ps, c_or_cg) for x, y, theta in zip(xs, ys, thetas)]
                distance_from_boundary = np.array([(x_max - x) / norm for x, x_max in zip(xs, x_max)])
                raise NotImplementedError


def stack_of_images(xs, ys, thetas, maze, d) -> np.array:
    imgs = []
    for i in range(len(xs)):
        maze.set_configuration([xs[i], ys[i]], thetas[i])
        maze.draw(display=d)
        img = d.get_image()
        # only keep pixels that are red [255, 0, 0]
        img_load = np.logical_and(img[:, :, 0] == 250, img[:, :, 1] == 0)

        binary_image = np.uint8(img_load)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_load = np.zeros_like(binary_image)
        img_load = cv2.drawContours(img_load.copy(), contours, 0, 1, thickness=cv2.FILLED)
        imgs.append(img_load)
        d.renew_screen()
    return np.stack(imgs)


def find_percentages(percentages_image, d, maze) -> list:
    # find the percentage that slide on the correct spot
    y2_up, y2_down, x2 = {'XL': [273, 410, 678],
                          'L': [277, 413, 695],
                          'M': [262, 390, 655],
                          'S': [174, 276, 624],
                          }[size]
    # cut out square from average_image around (y1, x1) with size 5
    cut_out_size = int(d.ppm * np.diff(maze.slits) / 20)
    percentages = []
    for y2 in [y2_up, y2_down]:
        cut_out = percentages_image[y2 - cut_out_size: y2 + cut_out_size, x2 - cut_out_size: x2 + cut_out_size]
        percentages.append(np.round(np.max(cut_out), 2).astype(float))
    return percentages


def plot_entire_maze_percentages(percentages_image, d, maze, key, condition, percentages):
    average_image = 1 - np.stack([percentages_image for _ in range(3)], axis=2)  # extend to 3 dimensions for plotting
    average_image_scaled = average_image / np.max(average_image)
    d.renew_screen()
    maze.set_configuration([-1, -1], 0)
    maze.draw(display=d)
    img = d.get_image()
    maze_boundary = img[:, :, 0] == 0
    average_image_scaled[maze_boundary] = [1, 0, 0]  # where maze_boundary is true, draw red on average_image
    plt.imshow(average_image_scaled)
    plt.gca().text(40, 40, percentages)
    plt.savefig(folder + 'single_exp_shift' + condition + '\\' + key.replace(': ', '_') + '_histogram.png')
    plt.close()


def plot_histogram(percentages_image):
    plt.hist(percentages_image.flatten(), bins=20)
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.savefig(folder + 'single_exp_shift' + condition + '\\' + key.replace(': ', '_') + '_values.png')
    plt.close()


def density_around_corner():
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    d = Display('', 1, maze)
    percentages = {}

    for i, key in tqdm(enumerate(all_coords_x.keys()), desc=size, total=len(all_coords_x.keys())):
        xs = all_coords_x[key]
        ys = all_coords_y[key]
        thetas = all_coords_theta[key]
        frames = all_coords_frames[key]

        if len(frames) > 60:
            imgs = stack_of_images(xs, ys, thetas, maze, d)
            percentages_image = np.sum(imgs, axis=0) / imgs.shape[0]
            percentages[key] = find_percentages(percentages_image, d, maze)

            plot_histogram(percentages_image)
            plot_entire_maze_percentages(percentages_image, d, maze, key, condition, percentages)

    # save percentages in json file with name 'folder + 'single_exp_shift' + condition + '\\size'
    with open(folder + 'single_exp_shift' + condition + '\\' + size + '_percentages.json', 'w') as file:
        json.dump(percentages, file)

    print(size)
    print(percentages.values())


def plot_excel_sheet_values(name: str):
    fit_results = pd.read_excel(folder + 'fit_results_' + name + '.xlsx', index_col=0)
    plt.errorbar(fit_results.index, fit_results['a'], yerr=fit_results['a_err'], fmt='o', label='a')
    plt.errorbar(fit_results.index, fit_results['b'], yerr=fit_results['b_err'], fmt='o', label='b')
    plt.legend()
    # write the function exp
    if name == 'power_law':
        plt.title('Fit results: a*x^b')
    if name == 'exp':
        plt.title('Fit results: a*exp(-b*x)')
    plt.savefig(folder + 'fit_results_' + name + '.png')
    plt.close()


def correct_trajectory(traj, space, inside, ps):
    # ps.visualize_space(space=space)
    # ps.draw(positions=traj.position[inside], angles=traj.angle[inside], color=(1, 0, 0), scale_factor=0.02)
    # ps.draw(positions=traj.position[~inside], angles=traj.angle[~inside], color=(0, 1, 0), scale_factor=0.02)
    DEBUG = 1


def correct_trajectory1(traj, space, inside, ps):
    # plot space
    plt.figure()
    space_no_y = np.any(space, axis=1)
    axim = plt.imshow(space_no_y.T,
                      aspect='equal',
                      cmap='Greys', vmin=0, vmax=3,
                      extent=[0, ps.indices_to_coords(ps.space.shape[0], 0, 0)[0],
                              0, ps.indices_to_coords(0, 0, ps.space.shape[2])[2]])
    # add on top of the image the scatter plot with markersize 2
    axim.axes.scatter(traj.position[inside, 0], traj.angle[inside],
                      c='k', s=10, marker='o', alpha=0.5, label='inside')
    axim.axes.scatter(traj.position[~inside, 0], traj.angle[~inside],
                      c='r', s=10, marker='o', alpha=0.5, label='outside')
    axim.axes.legend()
    plt.show(block=False)

    plt.figure()
    space_no_theta = np.any(space, axis=2)
    axim = plt.imshow(space_no_theta.T,
                      aspect='equal',
                      cmap='Greys', vmin=0, vmax=3,
                      extent=[0, ps.indices_to_coords(ps.space.shape[0], 0, 0)[0],
                              0, ps.indices_to_coords(0, ps.space.shape[1], 0)[1]])
    # add on top of the image the scatter plot with markersize 2
    axim.axes.scatter(traj.position[inside, 0], traj.position[inside, 1],
                      c='k', s=10, marker='o', alpha=0.5, label='inside')
    axim.axes.scatter(traj.position[~inside, 0], traj.position[~inside, 1],
                      c='r', s=10, marker='o', alpha=0.5, label='outside')
    axim.axes.legend()
    plt.show(block=False)
    # show image
    DEBUG = 1

    return traj


if __name__ == '__main__':
    folder = 'results\\distances_from_boundary\\'
    # name = 'power_law'
    name = 'exp'
    condition = '_x'

    # '_x', ''
    fit_results = pd.DataFrame(columns=['a', 'b', 'a_err', 'b_err'])
    fit_results.to_excel(folder + 'fit_results_' + name + '.xlsx')

    # # ________________load coordinates _______________________
    #
    # # save_coords()
    #
    for size in ['S', 'M', 'XL', 'L', ]:
        with open(folder + size + 'all_coords_x.json', 'r') as json_file:
            all_coords_x = json.load(json_file)
            json_file.close()

        with open(folder + size + 'all_coords_y.json', 'r') as json_file:
            all_coords_y = json.load(json_file)
            json_file.close()

        with open(folder + size + 'all_coords_theta.json', 'r') as json_file:
            all_coords_theta = json.load(json_file)
            json_file.close()

        with open(folder + size + 'all_coords_frames.json', 'r') as json_file:
            all_coords_frames = json.load(json_file)
            json_file.close()

        # ________________calculate calc_distances _______________________
        # filenames = {'L_SPT_4660001_LSpecialT_1_ants (part 1)': 50041,
        #              'XL_SPT_4630012_XLSpecialT_1_ants (part 1)': 61275,
        #              'S_SPT_5180005_SSpecialT_1_ants (part 1)': 17436,
        #              'M_SPT_4690010_MSpecialT_1_ants': 5722,
        #              }
        #
        # keys = [filename + ': ' + str(frame) for filename, frame in filenames.items()]
        #
        keys = list(all_coords_x.keys())

        # reduce all_coords to the keys
        all_coords_x = {key: all_coords_x[key] for key in keys if key in all_coords_x.keys()}
        all_coords_y = {key: all_coords_y[key] for key in keys if key in all_coords_y.keys()}
        all_coords_theta = {key: all_coords_theta[key] for key in keys if key in all_coords_theta.keys()}
        all_coords_frames = {key: all_coords_frames[key] for key in keys if key in all_coords_frames.keys()}

        if len(all_coords_x):
            find_distances(size, all_coords_x, all_coords_y, all_coords_theta, all_coords_frames, condition='_x')

# fig, ax = plt.subplots(figsize=(5, 5))
# for size in ['XL', 'L', 'M', 'S']:
#     with open(folder + size + 'values' + condition + '.json', 'r') as json_file:
#         values = json.load(json_file)
#         json_file.close()
#
#     with open(folder + size + 'bins' + condition + '.json', 'r') as json_file:
#         bins = np.array(json.load(json_file))
#         json_file.close()
#
#     # normalize histogram values
#     values = np.array(values)
#     y_values = values / np.sum(values)
#
#     dx = bins[1] - bins[0]
#     x_values = bins[:-1]/dx
#     ax.plot(x_values, y_values, '-o', label=size, markersize=3)

# set minimal xlim to 0
# plt.xlim(left=-2)

plt.xlabel('Distance from boundary (normalized)' + condition)
plt.ylabel('Probability')
plt.legend()

DEBUG = 1
plt.savefig(folder + 'distance_from_boundary' + condition + '.png')
plt.close()

# # ________________fit function _______________________
# for size in ['XL', 'L', 'M', 'S']:
#     with open(folder + size + 'values' + condition + '.json', 'r') as json_file:
#         values = json.load(json_file)
#         json_file.close()
#
#     with open(folder + size + 'bins' + condition + '.json', 'r') as json_file:
#         bins = json.load(json_file)
#         json_file.close()
#
#     fit_function_to_histogram(values, bins, eval(name), name)
#
# plot_excel_sheet_values(name)

# what could be reasons that Small has really off values?
# I need to find a way to shift the values in x to the left, so that the values are closer to the boundary

DEBUG = 1
