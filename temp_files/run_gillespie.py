from trajectory_inheritance.trajectory_gillespie import TrajectoryGillespie

x = TrajectoryGillespie(size='XL', shape='SPT', filename='gillespie_trajectory1')
x.setup_simulation()
x.run_simulation(frameNumber=5000)

# x.save()
#
# from trajectory_inheritance.get import get
# x = get('gillespie_trajectory1', 'gillespie')
# x.play()
