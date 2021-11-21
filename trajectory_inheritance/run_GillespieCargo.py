
from trajectory_inheritance.trj_GillespieCargo import Trajectory_gillespie

x = Trajectory_gillespie(size='L', shape='SPT', filename='gillespie_trajectory1')
x.run_simulation(frameNumber=30)

# x.save()
#
# from trajectory_inheritance.trajectory import get
# x = get('gillespie_trajectory1', 'gillespie')
# x.play()

