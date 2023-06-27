import json
import pandas as pd

from trajectory_inheritance.get import get
from DataFrame.import_excel_dfs import df_ant, df_human, df_pheidole
from tqdm import tqdm

# concat all df
df = pd.concat([df_ant, df_human, df_pheidole])

frameNum_dict = {}
# iterate over every row in df
for index, row in tqdm(df.iterrows()):
    # get filename
    filename = row['filename']
    # get solver
    solver = row['solver']

    x = get(filename)
    frameNum_dict[filename] = len(x.frames)
    # if row['time [s]'] != x.timer():
    #     print(row['time [s]']/x.timer())

# save in json
with open('frameNum_dict.json', 'w') as json_file:
    json.dump(frameNum_dict, json_file)
    json_file.close()