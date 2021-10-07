from Load_tracked_data.Load_Experiment import Load_Experiment
from Directories import MatlabFolder
from os import listdir

# StartedScripts: You still have to load the SL Asymmetric H. Its difficult, because the maze is twisted.
#  I have to untwist the data
# StartedScripts: Load the new human and ant movies

for mat_filename in listdir(MatlabFolder('ant', 'SL', 'RASH', False)):
    print(mat_filename)
    if 'part' not in mat_filename:

        x1 = Load_Experiment('ant', mat_filename,
                             [], True, [+0.2], [+0.2], [0], 50,
                             False, shape='LASH', size='SL')

        # x2_con_x3 = SmoothConnector(x2, x3)
        #
        # z = x1 \
        # + SmoothConnector(x1, x2) + x2 \
        # + SmoothConnector(x2, x3) + x3

        x1.play(4, 'Display')
        k = 2
        x1.save()
