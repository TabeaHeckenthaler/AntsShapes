from PhaseSpaces import PhaseSpace


solver = 'human'
sizes = ['Large', 'L', 'M', 'S']
shape = 'SPT'
new2021 = True

for size in sizes:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='', new2021=new2021)
    conf_space.load_space(new2021=new2021)
    # conf_space.save_space()
    conf_space.visualize_space()

    DEBUG = 1
