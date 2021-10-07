from trajectory_inheritance.trajectory import get
from PhysicsEngine.Contact import Contact_analyzer
from classes.bundle import Bundle

""" access single trajectories """
x = Get('XL_I_4190001_50fpsXL_I_14_ants', 'ant' )
x, contact = x.play(5, 'Display', 'contact')
# x.plot(5)


''' access all the trajectories of one kind '''
my_bundle = Bundle(size='XS', shape='H')
for x in list(my_bundle):
    print(x.position[1])


''' Example of how you can '''
x = get('XL_I_4190003_50fpsXL_I_12_ants', 'ant')
x.play(2, 'Display', 'contact')
omega, torque, first_contact = Contact_analyzer(x)
