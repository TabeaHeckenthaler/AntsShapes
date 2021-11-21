from PhaseSpaces import PhaseSpace
import os
from Directories import PhaseSpaceDirectory
import itertools
from joblib import Parallel, delayed

solvers = ['ant', 'human', 'humanhand']
shapes = ['SPT', 'H', 'I', 'T', ]
sizes = ['XL']
point_particle_bools = [False, True, ]


def calc(point_particle, shape, size, solver, parallel=False):
    name = size + '_' + shape

    if point_particle:
        name = name + '_pp'

    ps = PhaseSpace.PhaseSpace(solver, size, shape, name=name)
    ps.load_space(point_particle=point_particle)  # here, I used to have parallel...

    # ps.save_space(path=os.path.join(ps_dir, solver, name + ".pkl"))
    # ps.visualize_space(ps.name)
    print('Done: ' + name + ' ' + solver)
    return ps


if __name__ == '__main__':
    results = Parallel(n_jobs=5)(delayed(calc)(point_particle, shape, size, solver)
                                 for point_particle, shape, size, solver in
                                 itertools.product(point_particle_bools, shapes, sizes, solvers)
                                 )

    # calc(True, 'H', 'XL', parallel=False)
    # calc(True, 'I', 'XL', parallel=False)
