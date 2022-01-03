from PhaseSpaces import PhaseSpace

# sizes = ['Small Far', 'Small Near', 'Large', 'Medium']
sizes = ['L', 'M', 'S']  # TODO: There ones are HUGE. Let them run over night
shape = 'SPT'
solver = 'ant'

if __name__ == '__main__':

    for size in sizes:
        conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=True)
        conf_space.load_space()
        # conf_space.visualize_space()
        # conf_space.save_space(path=size + '_' + shape + '.pkl')

