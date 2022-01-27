from trajectory_inheritance.trajectory import get
from Load_tracked_data.Load_Experiment import load, connector, parts, time_dict, winner_dict
import numpy as np
import os


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
        x = load(mat_filename, solver, size, shape)
        chain = [x] + [load(filename, solver, size, shape, winner=x.winner)
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


if __name__ == '__main__':
    to_interpolate()
