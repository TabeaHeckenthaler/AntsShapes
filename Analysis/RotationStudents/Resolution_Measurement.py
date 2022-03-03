from trajectory_inheritance.trajectory import get
from Analysis.PathLength import PathLength
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
import json
from DataFrame.dataFrame import get_filenames
from tqdm import tqdm


def velocity(position: np.array, fps: int) -> np.array:
    v = np.array([])
    n = len(position) - 1
    for y in range(n):
        r = position[y + 1] - position[y]
        g = np.sqrt(r[0] ** 2 + r[1] ** 2) * fps
        v = np.append(v, g)
    return v


def smooth_v(v) -> np.array:
    """
    Smooth the v using np.medfilt (smoothing window ~30)
    :param v: velocity
    :return: smoothed velocity
    """
    return medfilt(v, 31)


def calculate_v_max(v) -> float:
    """
    :param v: velocity
    :return: maximal velocity
    """
    return np.max(v)


def acceleration_frames(v, limiter, cutoff) -> list:
    """

    """
    v_max = calculate_v_max(smoothed_v)

    for f2 in np.where(v >= limiter * v_max)[0]:
        f1s = [frame for frame, v in enumerate(v) if v <= cutoff * v_max and frame < f2]
        if len(f1s) > 0:
            f1 = np.max(f1s)
            return [f1, f2]
    return []


def plot_acceleration_frames(frames, marker='s'):
    plt.plot(x_axis[frames[0]], smoothed_v[frames[0]], "r" + marker)
    plt.plot(x_axis[frames[1]], smoothed_v[frames[1]], "g" + marker)


get_filenames('human', size='', shape='', free=False)


if __name__ == '__main__':
    shape, solver, geometry = 'SPT', 'human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')  # tuple
    resolution_dict = dict()

    for size in ['Large', 'Medium', 'Small Far']:
        path_list = []
        for name in tqdm(get_filenames('human', size=size, shape='SPT', free=False)):
            trajectory = get(name)

            smoothed_v = smooth_v(velocity(trajectory.position, trajectory.fps))
            frames = acceleration_frames(smoothed_v, 0.25, 0.05)
            if len(frames) > 1:
                path_list.append(PathLength(trajectory).calculate_path_length(frames=frames, rot=False))

                if path_list[-1] > 5:
                    x_axis = np.array(range(len(trajectory.position) - 1)) / trajectory.fps
                    plt.plot(x_axis, smoothed_v, color='blue')
                    plot_acceleration_frames(frames, marker='s')
                    plt.show()

        resolution_dict[size] = (np.mean(path_list), np.std(path_list), np.std(path_list) / np.sqrt(np.size(path_list)))
    print(resolution_dict)

    plt.figure()
    plt.errorbar([mean[0] for mean in resolution_dict.values()], [sem[2] for sem in resolution_dict.values()])
    plt.show()

    with open('bin_size.json', 'w') as fp:
        json.dump(resolution_dict, fp)
