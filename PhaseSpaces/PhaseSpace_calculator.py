from PhaseSpaces import PhaseSpace


solver = 'human'
sizes = ['Small Far']
# sizes = ['Large', 'Medium', 'Small Far']
shape = 'SPT'
new2021 = True

for size in sizes:
    conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(solver, size, shape, new2021=new2021)
    conf_space_labeled.load_eroded_labeled_space(new2021=new2021)
    # conf_space_labeled.visualize_states(reduction=10)
    # conf_space_labeled.visualize_transitions(reduction=10)

    DEBUG = 1
