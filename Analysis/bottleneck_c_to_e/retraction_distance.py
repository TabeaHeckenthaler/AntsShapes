from Analysis.bottleneck_c_to_e.correlation_edge_walk_decision_c_e_ac import *
from Setup.Maze import Maze
from ConfigSpace.ConfigSpace_SelectedStates import ConfigSpace_SelectedStates


# I want to quantify how strongly the vel direction before and after collision with the wall are correlated.
# Or better said: How far does the shape walk in the same direction though it is in contact with the wall?


# We have to quantify what it means to walk in the same direction.
# every time point has a three dimensional velocity vector.

def find_largest_x(x, y, theta, ps, c_or_cg):
    ind = ps.coords_to_indices(0, y, theta)
    available_x_ind = np.where(c_or_cg[:, ind[1], ind[2]])[0]
    if len(available_x_ind) == 0:
        return x
    max_x_ind = max(available_x_ind)
    max_x = ps.indices_to_coords(max_x_ind, ind[1], ind[2])[0]
    return max_x


def quantify_retraction(all_coords_x, all_coords_y, all_coords_theta):
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    norm = maze.slits[1] - maze.slits[0]
    # plot_2d_density(np.concatenatSe(all_coords_x), np.concatenate(all_coords_y), maze=maze, title=str(size))

    # write a function that returns for every set of y and theta a number in x,
    # which is the largest permissible x.
    ps = ConfigSpace_SelectedStates(solver=maze.solver, size=maze.size, shape=maze.shape,
                                    geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                              'LoadDimensions_new2021_SPT_ant.xlsx'))
    ps.load_final_labeled_space()
    c_or_cg = np.logical_or(ps.space_labeled == 'c', ps.space_labeled == 'cg')

    for xs, ys, thetas, i in zip(all_coords_x, all_coords_y, all_coords_theta, range(len(all_coords_x))):
        if len(xs) > 50:
            x_max = [find_largest_x(x, y, theta, ps, c_or_cg) for x, y, theta in zip(xs, ys, thetas)]
            distance_x_from_boundary = np.array([(x_max - x) / norm for x, x_max in zip(xs, x_max)])
            values, bins, bar_container = plt.hist(distance_x_from_boundary, bins=100, density=True)
            plt.close()
            # find the bin with the maximal number of counts
            max_bin = bins[np.argmax(values)]
            distance_x_from_boundary -= max_bin

            d_corrected = distance_x_from_boundary.copy()

            # find d_corrected < 0.05
            close_to_wall = d_corrected < 0.05

            # make all values in distance_x_from_boundary which are negative zero
            # d_corrected[d_corrected < 0] = 0

            DEBUG = 1

            plt.figure()
            plt.plot(distance_x_from_boundary, label='original')
            plt.plot(d_corrected, label='corrected')

            # draw red stripes where the shape is close to the wall
            for j in range(len(close_to_wall)):
                if close_to_wall[j]:
                    plt.axvspan(j, j + 1, facecolor='r', alpha=0.2)

            plt.xlabel('time')
            plt.ylabel('distance from boundary in x norm')
            plt.title(str(size) + ' ' + str(i))
            plt.legend()
            plt.savefig('results\\distances_in_x_from_boundary\\single_exp_retraction\\' + size + '_retraction_distance_x_' + str(i) + '.png')
            plt.close()


if __name__ == '__main__':
    folder = 'results\\distances_in_x_from_boundary\\'
    for size in ['XL', 'L', 'M', 'S']:
        with open(folder + size + 'all_coords_x.json', 'r') as json_file:
            all_coords_x = json.load(json_file)
            json_file.close()

        with open(folder + size + 'all_coords_y.json', 'r') as json_file:
            all_coords_y = json.load(json_file)
            json_file.close()

        with open(folder + size + 'all_coords_theta.json', 'r') as json_file:
            all_coords_theta = json.load(json_file)
            json_file.close()

        quantify_retraction(all_coords_x, all_coords_y, all_coords_theta)
