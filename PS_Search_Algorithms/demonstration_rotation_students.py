from PS_Search_Algorithms.Path_planning_full_knowledge import run_full_knowledge

for size in ['XL', 'L', 'M', 'S']:
    x = run_full_knowledge(size=size, shape='SPT', solver='ps_simulation',
                           geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                           initial_cond='front')

    x.play(wait=10)
