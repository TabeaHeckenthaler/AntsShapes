from trajectory_inheritance.trajectory import get
from Load_tracked_data.Load_Experiment import load, connector, parts, time_dict
import numpy as np
import os
from Setup.Maze import Maze
from trajectory_inheritance.trajectory_ant import Trajectory_ant


def to_interpolate():
    interpolate_list = {
        'S_SPT_4720005_SSpecialT_1_ants (part 1)': [],
    }
    easy_interpolate_list = {
        'S_SPT_4720005_SSpecialT_1_ants (part 1)': [],
    }

    for filename, exclude in interpolate_list.items():
        x = get(filename)
        print(x)
        new = x.easy_interpolate(easy_interpolate_list[filename])
        new = new.interpolate(exclude)
        new.play(frames=[0, -1])
        new.save()


def faulty_connection():
    solver, shape = 'ant', 'SPT'
    conn_dict = []

    for mat_filename, size in zip(conn_dict, ['S' for _ in range(3)]):
        x = load(mat_filename, solver, size, shape, fps)
        chain = [x] + [load(filename, solver, size, shape, fps, winner=x.winner)
                       for filename in parts(mat_filename, solver, size, shape)[1:]]
        total_time_seconds = np.sum([traj.timer() for traj in chain])

        frames_missing = (time_dict[mat_filename] - total_time_seconds) * x.fps

        for part in chain[1:]:
            frames_missing_per_movie = int(frames_missing / (len(chain) - 1))
            if frames_missing_per_movie > 10 * x.fps:
                connection = connector(x, part, frames_missing_per_movie)
                x = x + connection
            x = x + part
        x.play(step=5)
        x.save()


def delete_from_file(deletable_line):
    with open("check_trajectories.txt", "r") as input_:
        with open("temp.txt", "w") as output_:
            for line in input_:
                if line.strip("\n").split(':')[0] != deletable_line:
                    output_.write(line)
    os.replace('temp.txt', 'check_trajectories.txt')


def play_trajectories():
    file1 = open('check_trajectories.txt', 'r')
    trajectories_filenames = [line.split(':')[0].replace('\n', '')
                              for line in file1.readlines() if len(line) > 5]
    file1.close()

    start = 18
    for i, trajectories_filename in enumerate(trajectories_filenames[start:], start=start):
        print(trajectories_filename)
        print(i)
        x = get(trajectories_filename)
        x.play(step=1)

        if bool(int(input('OK? '))):
            delete_from_file(trajectories_filename)


def test_configuration(x):
    mymaze = Maze(x)
    mymaze.set_configuration([15, 4.912207], 0)
    mymaze.draw()


def extend_trajectory():
    extend_list = {
        # 'L_SPT_4080033_SpecialT_1_ants (part 1)': {'configuration': ([15, 4.912207], 0), 'where': 'beginning'},
        # 'L_SPT_4090010_SpecialT_1_ants (part 1)': {'configuration': ([15, 4.912207], 0), 'where': 'beginning'}
    }
    for filename, info in extend_list.items():
        x_cut = get(filename)
        x_extend = Trajectory_ant(size=x_cut.size,
                                  shape=x_cut.shape,
                                  old_filename=filename + '_extension',
                                  winner=x_cut.winner)

        x_extend.position = np.array([info['configuration'][0]])
        x_extend.angle = np.array([info['configuration'][1]])
        x_extend.frames = np.array([i for i in range(x_extend.position.shape[0])])

        if info['where'] == 'end':
            connection = connector(x_cut, x_extend, x_cut.fps * 10)
            x = x_cut + connection
        elif info['where'] == 'beginning':
            connection = connector(x_extend, x_cut, x_cut.fps * 10)
            x = connection + x_cut
        DEBUG = 1


if __name__ == '__main__':
    extend_trajectory()
