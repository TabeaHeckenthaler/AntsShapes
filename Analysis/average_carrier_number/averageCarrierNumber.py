from DataFrame.dataFrame import myDataFrame
import json
from Directories import averageCarrierNumber_dir


def create():
    averageCarrierNumber = {f: N for f, N in zip(myDataFrame['filename'], myDataFrame['average Carrier Number'])}

    with open(averageCarrierNumber_dir, 'w') as json_file:
        json.dump(averageCarrierNumber, json_file)
        json_file.close()


with open(averageCarrierNumber_dir, 'r') as json_file:
    averageCarrierNumber = json.load(json_file)


myDataFrame['average Carrier Number'] = myDataFrame['filename'].map(averageCarrierNumber)
DEBUG = 1