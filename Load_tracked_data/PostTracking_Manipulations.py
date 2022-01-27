from trajectory_inheritance.trajectory import get
from Load_tracked_data.Load_Experiment import load, connector, parts, time_dict, winner_dict
import numpy as np
import json

# TODO: Correct these.


def to_interpolate():
    interpolate_list = {
        'S_SPT_4720005_SSpecialT_1_ants (part 1)': [],
        # cut off at 49704
    }

    for filename, exclude in interpolate_list.items():
        x = get(filename)
        print(x)
        # new = x.easy_interpolate([[28471, 31179]])
        # new = new.interpolate(exclude)
        x.play(frames=[0, -1])
        x.save()


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


if __name__ == '__main__':
    to_interpolate()
