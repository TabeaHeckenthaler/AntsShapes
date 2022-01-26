from trajectory_inheritance.trajectory import get
from Load_tracked_data.Load_Experiment import load, connector, parts, time_dict, winner_dict
import numpy as np
import json


def to_interpolate():
    interpolate_list = {
        'S_SPT_4750014_SSpecialT_1_ants (part 1)': [[29250, 29707], [29778, 29790], [29827, 30004]],
        'S_SPT_4720014_SSpecialT_1_ants': [[20308, 20485], [20635, 20660]],
        'S_SPT_4750016_SSpecialT_1_ants': [[66, 77], [23851, 24428], [28144, 28167], [28472, 31180]],
        # cut off at 49704
        'S_SPT_4770012_SSpecialT_1_ants (part 1)': [[2498, 3190]],
        'S_SPT_4780002_SSpecialT_1_ants': [[28398, 28414]],
        'S_SPT_4790005_SSpecialT_1_ants (part 1)': [[-800, -700]]}

    for filename, exclude in interpolate_list.items():
        x = get(filename)
        print(x)
        new = x.interpolate(exclude)
        new.play()
        new.save()


def faulty_connection():
    solver, shape = 'ant', 'SPT'
    conn_dict = ['XLSPT_4640024_XLSpecialT_1_ants (part 1).mat',
                 'SSPT_4760017_SSpecialT_1_ants (part 1).mat',
                 'SSPT_4770007_SSpecialT_1_ants (part 1).mat',
                 'SSPT_4780009_SSpecialT_1_ants (part 1).mat']

    for mat_filename, size in zip(conn_dict, ['XL' for _ in range(1)] + ['S' for _ in range(3)]):
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
    faulty_connection()
