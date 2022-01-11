from PhaseSpaces import PhaseSpace


solver = 'human'
sizes = ['Large']
# sizes = ['Large', 'Medium', 'Small Far']
shape = 'SPT'
new2021 = True

for size in sizes:
    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(solver, size, shape, new2021=new2021)
    conf_space_labeled.load_eroded_labeled_space(new2021=new2021)
    # conf_space_labeled.save_space()
    # conf_space.visualize_space()

    DEBUG = 1
