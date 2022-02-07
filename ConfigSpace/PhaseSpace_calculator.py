from ConfigSpace import ConfigSpace_Maze
from trajectory_inheritance.exp_types import exp_types

shape = 'SPT'
solver = 'human'
sizes = ['Small Far'] # exp_types[shape][solver]

for size in sizes:
    conf_space = ConfigSpace_Maze.ConfigSpace_Maze(solver, size, shape, ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
    conf_space.load_space()
    conf_space.visualize_space()
    # conf_space.calculate_space()
    # conf_space.calculate_boundary()
    # conf_space.save_space()
    # conf_space_labeled.visualize_states(reduction=10)
    # conf_space_labeled.visualize_transitions(reduction=10)

    DEBUG = 1

