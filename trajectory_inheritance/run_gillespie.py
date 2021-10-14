
from trajectory_inheritance.trajectory_gillespie import Trajectory_gillespie

x = Trajectory_gillespie(size='L', shape='T', filename='gillespie_trajectory1')
x.run_simulation(frameNumber=5000)

x.save()
#
# from trajectory_inheritance.trajectory import get
# x = get('gillespie_trajectory1', 'gillespie')
# x.play()
