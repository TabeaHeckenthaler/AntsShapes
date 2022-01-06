from PhaseSpaces import PhaseSpace
#
# sizes = ['S', 'M', 'L']
# shape = 'SPT'
# solver = 'ant'
#
# if __name__ == '__main__':
#
#     for size in sizes:
#         conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name=size + '_' + shape, new2021=True)
#         conf_space.load_space()
#         # conf_space.visualize_space()
#         # conf_space.save_space(path=size + '_' + shape + '.pkl')

solver = 'ant'
sizes = ['L', 'M', 'S']
shape = 'SPT'
new2021 = True

for size in sizes:
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='', new2021=new2021)
    conf_space.load_space(new2021=new2021)
    # conf_space.save_space()
    conf_space.visualize_space()

    DEBUG = 1
