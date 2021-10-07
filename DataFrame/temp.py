import pandas as pd
from Directories import data_home
from trajectory_inheritance.trajectory import get
import numpy as np
from DataFrame.create_dataframe import get_filenames

df_dir = data_home + 'DataFrame\\data_frame'

solver = 'ps_simulation'
df = pd.read_json(df_dir + '.json')

df_dstar = df[df['solver'] == solver].copy()

for filename in get_filenames(solver):

    x = get(filename, solver)
    if not hasattr(x, 'frames') or x.frames.size == 0:
        x.frames = np.array([i for i in range(x.position.shape[0])])
        x.save()
        print(x.filename)
