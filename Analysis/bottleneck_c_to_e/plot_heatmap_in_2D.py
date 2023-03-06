from Analysis.bottleneck_c_to_e.correlation_edge_walk_decision_c_e_ac import *
from Setup.Maze import Maze


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
        f_title = {'XL': 1, 'L': 1/2, 'M': 1/4, 'S (> 1)': 1/8, 'Single (1)': 1/8}
        f = f_title[title]
        r = [[13 * f, 19 * f], [0 * f, 19.1 * f]]
    else:
        r = [[maze.slits[0], maze.slits[1]], [0, maze.arena_height]]

    h, x_edges, y_edges, image = plt.hist2d(x, y, bins=100, cmap='viridis', range=r)
    # set colorscale log
    h = np.log(h)

    ax = plt.imshow(h)
    ax.axes.set_title(title)
    # equal aspect ratio
    ax.axes.set_aspect('equal')

    # replace in the title the characters that are not allowed in a filename
    title = title_for_saving(title)

    # save the figure without the white border with high resolution
    plt.gcf().savefig('2d_density_log' + title + '.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # trace_scatter = go.Scatter(x=x, y=y, mode='markers')
    # trace_density = go.Densitymapbox(
    #     lat=y, lon=x, z=[1] * len(x),  # use a constant value for all points
    #     radius=5,  # set the radius of the points
    # )
    # layout = go.Layout(
    #     title=title,
    #     width=800,  # set the width of the plot to 800 pixels
    #     height=1200,  # set the height of the plot to 800 pixel
    #     mapbox=dict(
    #         center=dict(lat=np.mean(y), lon=np.mean(x)),
    #         style='white-bg',
    #         zoom=4),
    #     paper_bgcolor='rgba(0,0,0,0)',  # make the plot background transparent
    #     plot_bgcolor='rgba(0,0,0,0)'  # make the plot background transparent
    # )
    # fig = go.Figure(data=[trace_density, trace_scatter], layout=layout)
    # fig.show()
    # # fig.write_image('2d_scatter_density.svg')
    # save_fig(fig, '2d_density' + title)


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    for size, df in [(size, df) for size, df in dfs_ant.items() if size not in ['Single (1)']]:
        all_coords_x = []
        all_coords_y = []
        for filename in tqdm(df['filename']):
            # maze = Maze(get(df['filename'].iloc[0]))

            traj = get(filename)

            if traj.solver == 'ant' and \
                    traj.geometry() != ('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'):
                traj = traj.confine_to_new_dimensions()
            traj.smooth()

            ts = time_series_dict[filename]
            in_c_trajs, out_c_trajs = In_the_bottle.cut_traj(traj, ts, buffer=2)

            for in_c_traj in in_c_trajs:
                coord_x = in_c_trajs[0].position[:, 0][::in_c_traj.fps]
                coord_y = in_c_trajs[0].position[:, 1][::in_c_traj.fps]

                all_coords_x.append(coord_x)
                all_coords_y.append(coord_y)

        maze = Maze(get(df['filename'].iloc[0]))

        plot_2d_density(np.concatenate(all_coords_x), np.concatenate(all_coords_y), maze=maze, title=str(size))
