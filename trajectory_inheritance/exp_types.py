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

ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'human': {'Small': 0.25, 'Small Near': 0.25, 'Small Far': 0.25, 'Medium': 0.5, 'Large': 1},
                 'humanhand': {'': 1},
                 'gillespie': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 }


def is_exp_valid(shape, solver, size):
    error_msg = 'Shape ' + shape + ', Solver ' + solver + ', Size ' + size + ' is not valid.'
    if shape not in exp_types.keys():
        raise ValueError(error_msg)
    if solver not in exp_types[shape].keys():
        raise ValueError(error_msg)
    if size not in exp_types[shape][solver]:
        raise ValueError(error_msg)
