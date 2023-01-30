from trajectory_inheritance.get import get
from Directories import network_dir
import numpy as np
import os
import json
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from DataFrame.dataFrame import myDataFrame
from DataFrame.Altered_DataFrame import Altered_DataFrame
from trajectory_inheritance.exp_types import solver_geometry
from Analysis.Efficiency.PathLength import PathLength


time_step = 0.25
with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

solver = 'ant'

adf = Altered_DataFrame(myDataFrame)
adf.choose_experiments(solver=solver, shape='SPT', geometry=solver_geometry[solver], init_cond='back')
filenames = adf.df['filename']

area, p, area_per_distance = dict(), dict(), dict()
states = ['c', "cg"]

# filenames = ['medium_20210419142429_20210419142729', 'L_SPT_4660021_LSpecialT_1_ants (part 1)']

for filename in filenames:
    print(filename)
    x = get(filename)
    time_series = time_series_dict[filename]

    in_state = np.isin(np.array(['0'] + time_series + ['0']), states).astype(int)
    entered_exited = np.where(in_state[:-1] != in_state[1:])[0]
    if len(entered_exited) > 0:
        times = np.split(entered_exited, int(len(entered_exited) / 2))
    else:
        times = []
    area_per_distance[filename] = []

    for time in times:
        f = [int(time_step * time[0] * x.fps), int(time_step * time[1] * x.fps)]
        print(f)
        xdata = x.position[f[0]:f[1], 0]
        ydata = x.position[f[0]:f[1], 1]
        zdata = x.angle[f[0]:f[1]]

        # find complex hull of points
        hull = ConvexHull(np.array([xdata, ydata, zdata]).T)

        plt.figure()
        plt.plot(xdata, ydata, 'o')
        plt.plot(xdata[hull.vertices], ydata[hull.vertices], 'r--', lw=2)
        plt.plot(xdata[hull.vertices[0]], ydata[hull.vertices[0]], 'ro')
        plt.savefig('images\\ConvexHull\\' + filename + '.png')
        plt.close()

        area[filename] = hull.area
        p[filename] = PathLength(x).calculate_path_length(frames=time)

        area_per_distance[filename].append(area[filename] / p[filename]**2)

        # save area as txt file
        DEBUG = 1

    # with open(os.path.join('area_ants.txt'), 'w') as json_file:
    #     json.dump(area, json_file)
    #     json_file.close()
    #
    # with open(os.path.join('pathlength_c_ants.txt'), 'w') as json_file:
    #     json.dump(p, json_file)
    #     json_file.close()

    with open(os.path.join('area_per_distance_ants.txt'), 'w') as json_file:
        json.dump(area_per_distance, json_file)
        json_file.close()

DEBUG = 1
