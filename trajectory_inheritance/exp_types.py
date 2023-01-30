from Directories import home
from os import path
import json

centerOfMass_shift = - 0.08  # shift of the center of mass away from the center of the SpT load. # careful!

exp_types = {'SPT': {'ant': ['XL', 'L', 'M', 'S'],
                     'pheidole': ['XL', 'L', 'M', 'S'],
                     'human': ['Large', 'Medium', 'Small Far', 'Small Near'],
                     'ps_simulation': ['XL', 'L', 'M', 'S', 'Large', 'Medium', 'Small Far', 'Small Near', ''],
                     'humanhand': [''],
                     'gillespie': ['XL', 'L', 'M', 'S']},
             'H': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'I': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'T': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']}}

solver_geometry = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                   'humanhand': ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx'),
                   'gillespie': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'pheidole': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   }

with open(path.join(home, 'Setup', 'ResizeFactors.json'), "r") as read_content:
    ResizeFactors = json.load(read_content)

color = {'Large communication': '#cc3300',
         'Large non_communication': '#ff9966',
         'M (>7) communication': '#339900',
         'M (>7) non_communication': '#99cc33',
         'Small non_communication': '#0086ff',
         'XL': '#ff00c1',
         'L': '#9600ff',
         'M': '#4900ff',
         'S (> 1)': '#00b8ff',
         'Single (1)': '#00fff9',
         }

def is_exp_valid(shape, solver, size):
    error_msg = 'Shape ' + shape + ', Solver ' + solver + ', Size ' + size + ' is not valid.'
    if shape not in exp_types.keys():
        raise ValueError(error_msg)
    if solver not in exp_types[shape].keys():
        raise ValueError(error_msg)
    if size not in exp_types[shape][solver]:
        raise ValueError(error_msg)
