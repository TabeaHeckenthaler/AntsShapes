import pandas as pd
from os import listdir
from trajectory import home, solvers, SaverDirectories, Get, Save
import numpy as np
from DataFrame.create_dataframe import get_filenames

df_dir = home + 'DataFrame\\data_frame'

solver = 'dstar'
df = pd.read_json(df_dir + '.json')

df_dstar = df[df['solver'] == solver].copy()

for filename in get_filenames(solver):

    x = Get(filename, solver)
    if not hasattr(x, 'frames') or x.frames.size == 0:
        x.frames = np.array([i for i in range(x.position.shape[0])])
        Save(x)
        print(x.filename)
