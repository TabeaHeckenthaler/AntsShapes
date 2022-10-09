
from trajectory_inheritance.trj_GillespieCargo import TrajectoryGillespie

x = TrajectoryGillespie(size='XL', shape='SPT', filename='gillespie_trajectory1')
x.run_simulation(frameNumber=10000, display=True)
# x.play(step=10)
k = 1



# x.save()
#
# from trajectory_inheritance.get import get
# x = get('gillespie_trajectory1', 'gillespie')
# x.play()

