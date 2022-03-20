from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get


def choose_experiments(solver, shape, size, geometry, communication=None):
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
        (myDataFrame['initial condition'] == 'back') &
        (myDataFrame['maze dimensions'] == geometry[0]) &
        (myDataFrame['load dimensions'] == geometry[1])]

    if communication is not None:
        df = df[(df['communication'] == communication)]
    return df


def choose_trajectories(solver='human', size='Large', shape='SPT',
                     geometry: tuple = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'), number: int = None) \
        -> list:
    df = choose_experiments(solver, shape, size, geometry)
    filenames = df['filename'][:number]
    # filenames = ['XL_SPT_dil9_sensing' + str(ii) for ii in [5, 6, 7, 8, 9]]
    return [get(filename) for filename in filenames]