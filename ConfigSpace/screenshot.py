from ConfigSpace.ConfigSpace_SelectedStates import *
from PIL import Image
from mayavi import mlab

shape = 'SPT'
# geometries_to_change = {('humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')): ['']}

geometries = {
    ('human', ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')): ['Medium', 'Large', 'Small Far'],
    ('ant', ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx')): ['XL', 'L', 'M', 'S'],
    ('humanhand', ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')): ['']
}
for (solver, geometry), sizes in list(geometries.items()):
    for size in sizes:
        print(solver, size)

        cs_labeled = ConfigSpace_SelectedStates(solver, size, shape, geometry)
        cs_labeled.load_final_labeled_space()

        labels = np.unique(cs_labeled.space_labeled)
        for label in labels:
            cs_labeled.visualize_transitions(reduction=2, only_states=[label])
            cs_labeled.visualize_space(reduction=2)

            arr = mlab.screenshot(cs_labeled.fig, mode='rgb')
            arr = mlab.screenshot(cs_labeled.fig, mode='rgb')
            im = Image.fromarray(arr)
            im.save('images\\CS_check\\' + label + '\\' + solver + geometry[0].split('.')[0] + size + '.jpeg')
            mlab.close()

        DEBUG = 1
