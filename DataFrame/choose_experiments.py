from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get


def choose_experiments(solver, shape, size, geometry, communication=None, init_cond: str = 'back'):
    """
    Get trajectories based on the trajectories saved in myDataFrame.
    :param geometry: geometry of the maze, given by the names of the excel files with the dimensions
    :param number: How many trajectories do you want to have in your list?
    :param solver: str
    :param size: str
    :param shape: str
    :return: list of objects, that are of the class or subclass Trajectory
    """
    if 'Small' in size:
        sizes = ['Small Far', 'Small Near']
    else:
        sizes = [size]

    df = myDataFrame[
        (myDataFrame['size'].isin(sizes)) &
        (myDataFrame['shape'] == shape) &
        (myDataFrame['solver'] == solver) &
        (myDataFrame['initial condition'] == init_cond) &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    if communication is not None:
        df = df[(df['communication'] == communication)]
    return df


def choose_trajectories(solver='human', size='Large', shape='SPT',
                        geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                        communication=None, number: int = None, init_cond: str ='back') \
        -> list:
    df = choose_experiments(solver, shape, size, geometry, communication=communication, init_cond=init_cond)
    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    # bad_filename = 'S_SPT_4750002_SSpecialT_1_ants (part 1)' # TODO: fix this trajecotry
    return [get(filename) for filename in filenames]
