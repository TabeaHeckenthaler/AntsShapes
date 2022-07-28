from ConfigSpace import ConfigSpace_Maze
from trajectory_inheritance.exp_types import solver_geometry

shape = 'SPT'
solver = 'ant'
size = 'S'


conf_space = ConfigSpace_Maze.ConfigSpace_Maze(solver, size, shape, solver_geometry[solver])
conf_space.calculate_space()
conf_space.save_space()
# conf_space_labeled.visualize_states(reduction=10)
# conf_space_labeled.visualize_transitions(reduction=10)

DEBUG = 1

