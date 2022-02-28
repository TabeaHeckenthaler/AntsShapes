from PS_Search_Algorithms.Path_planning_full_knowledge import run_full_knowledge
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
import os

for size in ['XL', 'L', 'M', 'S']:
    conf_space = ConfigSpace_Maze(size=size, shape='SPT', solver='ps_simulation',
                                  geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                            'LoadDimensions_new2021_SPT_ant.xlsx'))
    if os.path.exists(conf_space.directory()):
        conf_space.load_space()
        conf_space.visualize_space()

    x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
                           geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                           initial_cond='front')
    x.play(wait=10, cs=conf_space)
