from PhaseSpaces import PhaseSpace

sizes = ['M', 'S']
solver = 'ant'
shape = 'H'

if __name__ == '__main__':
    for size in sizes:
        conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=True)
        conf_space.load_space()
        conf_space.save_space()
