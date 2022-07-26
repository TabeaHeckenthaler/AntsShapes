import json
from trajectory_inheritance.get import get
from trajectory_inheritance.trajectory_ant import Trajectory_ant

if __name__ == '__main__':
    # TODO: fix that ant traj are saved as simulations
    k_on, k_off = {}, {}
    experiments = {'XL': 'XL_SPT_4290009_XLSpecialT_2_ants',
                   'L': 'L_SPT_4080033_SpecialT_1_ants (part 1)',
                   'M': 'M_SPT_4680005_MSpecialT_1_ants',
                   'S': 'S_SPT_4800001_SSpecialT_1_ants (part 1)'}

    for size, filename in experiments.items():
        x = get(filename)
        x = Trajectory_ant(size=x.size, shape=x.shape, old_filename=x.old_filenames(0), free=False, fps=x.fps,
                           winner=x.winner, x_error=0, y_error=0, angle_error=0, falseTracking=x.falseTracking)
        x.load_participants()
        k_on[size] = x.participants.k_on(x.fps)
        k_off[size] = x.participants.k_off(x.fps)

    with open("k_on.json", "w") as outfile:
        json.dump(k_on, outfile)

    with open("k_off.json", "w") as outfile:
        json.dump(k_off, outfile)
