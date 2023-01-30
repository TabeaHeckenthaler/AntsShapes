import os
import json
import numpy as np

with open(os.path.join('area_per_distance_ants.txt'), 'r') as json_file:
    area_per_distance_ants = json.load(json_file)
    json_file.close()


with open(os.path.join('area_per_distance_humans.txt'), 'r') as json_file:
    area_per_distance_humans = json.load(json_file)
    json_file.close()

[a for a in area_per_distance_humans.values()]

flat_list_humans = [item for sublist in area_per_distance_humans.values() for item in sublist if not np.isinf(item)]
flat_list_ants = [item for sublist in area_per_distance_ants.values() for item in sublist if not np.isinf(item)]

print(np.median(flat_list_humans))
print(np.median(flat_list_ants))
j

DEBUG = 1
