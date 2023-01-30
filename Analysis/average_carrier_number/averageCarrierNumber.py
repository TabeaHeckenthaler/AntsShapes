from DataFrame.dataFrame import myDataFrame
import json
from Directories import averageCarrierNumber_dir
from trajectory_inheritance.get import get


def create_dict():
    """
    This only worked for the old dataFrame
    """
    averageCarrierNumber = {f: N for f, N in zip(myDataFrame['filename'], myDataFrame['average Carrier Number'])}

    with open(averageCarrierNumber_dir, 'w') as json_file:
        json.dump(averageCarrierNumber, json_file)
        json_file.close()


def extend_dictionary(df):
    missing_trajs = set(df[df['solver'] == 'ant']['filename']) - set(averageCarrierNumber_dict.keys())
    for filename in missing_trajs:
        x = get(filename)
        if 'SPT' in filename and not x.free and x.size == 'S':
            # a = x.averageCarrierNumber()
            # print(a)
            averageCarrierNumber_dict[filename] = 10
            DEBUG = 1

    with open(averageCarrierNumber_dir, 'w') as json_file:
        json.dump(averageCarrierNumber_dict, json_file)
        json_file.close()


with open(averageCarrierNumber_dir, 'r') as json_file:
    averageCarrierNumber_dict = json.load(json_file)


# extend_dictionary(myDataFrame)

# averageCarrierNumber_dict = {}
# myDataFrame['average Carrier Number'] = myDataFrame['filename'].map(averageCarrierNumber_dict)
# DEBUG = 1


# singles = myDataFrame[myDataFrame['solver'].isin(['pheidole', 'ant']) &
#                       (myDataFrame['average Carrier Number'] == 1) &
#                       (myDataFrame['size'] == 'S') &
#                       (myDataFrame['shape'] == 'SPT')].sort_values('filename')
#
# multiples = myDataFrame[myDataFrame['solver'].isin(['pheidole', 'ant']) &
#                         (myDataFrame['average Carrier Number'] > 1) &
#                         (myDataFrame['size'] == 'S') &
#                         (myDataFrame['shape'] == 'SPT')].sort_values('filename')
#
# singles.to_excel("singles.xlsx")
# multiples.to_excel("multiples.xlsx")
