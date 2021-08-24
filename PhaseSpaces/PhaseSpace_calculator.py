from PhaseSpaces import PhaseSpace
import os
from PhaseSpaces.PhaseSpace import ps_dir
import itertools
from joblib import Parallel, delayed


solver = 'ant'
shapes = ['H', 'I', 'T', 'SPT']
sizes = ['XL']
point_particle_bools = [False, True, ]


def calc(point_particle, shape, size):
    name = size + '_' + shape

    if point_particle:
        name = name + '_pp'

    path = os.path.join(ps_dir, solver, name + ".pkl")
    ps = PhaseSpace.PhaseSpace(solver, size, shape, name=name)
    ps.load_space(path=path, point_particle=point_particle)

    # ps.calculate_boundary(point_particle=point_particle)
    ps.save_space(path=os.path.join(ps_dir, solver, name + ".pkl"))
    # ps.visualize_space(ps.name)
    print(shape)


if __name__ == '__main__':
    Parallel(n_jobs=5)(delayed(calc)(point_particle, shape, size)
                       for point_particle, shape, size in
                       itertools.product(point_particle_bools, shapes, sizes)
                       )
    # calc(False, 'H', 'XL')
