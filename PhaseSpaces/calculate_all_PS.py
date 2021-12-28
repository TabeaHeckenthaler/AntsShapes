from PhaseSpaces import PhaseSpace

sizes = ['Large', 'Medium', 'Small Far']
shape = 'SPT'
solver = 'human'

if __name__ == '__main__':
    for size in sizes:
        conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=False)
        conf_space.load_space()
        conf_space.visualize_space()
        # conf_space.save_space(path=size + '_' + shape + '.pkl')


# from PhaseSpaces import PhaseSpace
#
# sizes = ['XS', 'S', 'M', 'L', 'SL', 'XL']
# shape = 'T'
# solver = 'ant'
#
# if __name__ == '__main__':
#     for size in sizes:
#         conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=True)
#         conf_space.load_space()
#         conf_space.save_space()


# from PhaseSpaces import PhaseSpace
#
# sizes = ['XS', 'S', 'M', 'L', 'SL', 'XL']
# shape = 'I'
# solver = 'ant'
#
# if __name__ == '__main__':
#     for size in sizes:
#         conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=True)
#         conf_space.load_space()
#         conf_space.save_space()