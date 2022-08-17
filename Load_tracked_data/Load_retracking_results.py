from Load_tracked_data.Load_Experiment import load
from trajectory_inheritance.get import get
from matplotlib import pyplot as plt


def split_trajectory(filename):
    x = get(filename)
    xs = x.divide_into_parts()

    x_new = xs[0]

    for chain_name, x_part in zip(x.VideoChain[1:], xs[1:]):
        if 'CONNECTOR' in chain_name:
            new_filenames = x.size + '_'.join((chain_name.split(')')[0] + 'r).mat').split('_')[1:])
            xs_loaded = load(new_filenames, x.solver, x.size, x.shape, x.fps, [], winner=x.winner)
            x_new = x_new + xs_loaded
        else:
            x_new = x_new + x_part

    print(x_new.VideoChain)
    print('time ', x_new.timer()/60, ' min')
    print('winner ', x_new.winner)
    x_new.free = False
    x_new.play(step=10)
    x_new.play(frames=[-5000, -1], step=1)
    # x_new1.play(frames=[54000, 56000], step=1, wait=20)

    D = 1
    # x_new.save()


if __name__ == '__main__':
    filenames = ["M_SPT_4690009_MSpecialT_1_ants (part 1)",
                 "S_SPT_4740008_SSpecialT_1_ants (part 1)",
                 "S_SPT_4730013_SSpecialT_1_ants (part 1)",
                 "S_SPT_4710014_SSpecialT_1_ants (part 1)",
                 "S_SPT_4720002_SSpecialT_1_ants (part 1)",
                 "S_SPT_4720005_SSpecialT_1_ants (part 1)",
                 "S_SPT_4740002_SSpecialT_1_ants (part 1)",
                 "S_SPT_4750009_SSpecialT_1_ants (part 1)",
                 "S_SPT_4750007_SSpecialT_1_ants (part 1)",
                 ]

    for filename in filenames[0:]:
        split_trajectory(filename)
