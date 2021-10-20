from matplotlib import pyplot as plt
from os import getcwd
import numpy as np
from Setup.Maze import Maze
from os import listdir, path, walk
from Analysis.GeneralFunctions import read_text_file
from Directories import SaverDirectories, trackedAntMovieDirectory, NewFileName
from trajectory_inheritance.trajectory import get
from Load_tracked_data.Load_Experiment import Load_Experiment

c = getcwd()
TextFileDirectory = path.join(SaverDirectories['ant'], 'Classes_Experiment', 'AssymetricH_Series')


def Zipper(lines):
    originalMovies = [lines[index].split(' ')[0] for index in range(1, len(lines)) if
                      lines[index].split(' ')[0].startswith('S')]
    Times = [lines[index].split(' ')[1][:5] for index in range(1, len(lines)) if
             ':' in lines[index].split(' ')[1] and lines[index].split(' ')[0].startswith('S')]
    return dict(zip(originalMovies, Times))


def runNumber(x):
    return x.filename.split('_')[-2]


def Movie(x):
    return x.filename.split('_')[2]


def get_sec(time_str, **kwargs):
    """Get Seconds from time."""
    h, m = time_str.split(':')
    return int(h) * 60 + int(m)


def firstMovies():
    return [name[:-4] for name in listdir(path.join(SaverDirectories['ant'], 'AssymetricH_Series'))]


class Series():
    def __init__(self, firstmovie):
        lines = read_text_file(TextFileDirectory, firstmovie + '.txt')
        self.directory = lines[0]
        self.originalMovies = [i for i in Zipper(lines).keys()]
        self.trajectories = []

        for originalMovie in [x.replace('S', '') for x in self.originalMovies]:
            for exp in listdir(getcwd() + path.sep + 'Ant_Trajectories'):
                if originalMovie in exp and ('ASH' in exp):
                    self.trajectories.append(get(exp, 'ant'))

    def __str__(self):
        return str([x.filename for x in self.trajectories])

    def __len__(self):
        return len(self.trajectories)

    def sort(self):
        self.trajectories.sort(key=runNumber)
        self.trajectories.sort(key=Movie)

    def Max(self, function, *args, **kwargs):
        valueList = [function(x, *args, **kwargs) for x in self.trajectories]
        return self.trajectories[np.where(valueList == max(valueList))[0][0]]

    def Min(self, function, *args, **kwargs):
        valueList = [function(x, *args, **kwargs) for x in self.trajectories]
        return self.trajectories[np.where(valueList == min(valueList))[0][0]]

    def StartingTimes(self):

        dic = Zipper(read_text_file(TextFileDirectory, 'S' + Movie(self.trajectories[0]) + '.txt'))
        for x in self.trajectories:
            x.time = dic['S' + Movie(self.trajectories[0])]

        StartingTimes = [x.time for x in self.trajectories]
        StartingTimes = [get_sec(t) for t in StartingTimes]
        return [x - StartingTimes[0] for x in StartingTimes]

    def plot(self, valueList, function_name, *args, **kwargs):
        from Analysis.GeneralFunctions import graph_dir

        plt.close()
        fig = plt.gcf()

        self.sort()

        valueListR, valueListL = [], []

        leftOrright = [x.shape[0] for x in self.trajectories]
        for index in range(len(leftOrright)):
            if leftOrright[index] == 'L':
                valueListR.append(np.NaN)
                valueListL.append(valueList[index])
            else:

                valueListR.append(valueList[index])
                valueListL.append(np.NaN)

        StartingTimes = self.StartingTimes()

        packs = []
        for originalMovie in [x.replace('S', '') for x in self.originalMovies]:
            packs = packs + [[]]
            for x in self.trajectories:
                packs[-1].append(originalMovie in x.filename)

        for pack in packs:
            for index in range(len(pack) - 1):
                if pack[index]:
                    StartingTimes[index + 1] = StartingTimes[index] + self.trajectories[index].timer() / 60

        plt.scatter(StartingTimes, valueListL, c='r', marker='*')
        plt.scatter(StartingTimes, valueListR, c='b', marker='*')

        plt.legend(['left', 'right'])
        plt.title('Assymetric H,  size = ' + self.trajectories[0].size + ', S' + Movie(self.trajectories[0]))
        plt.ylim(ymin=0)
        plt.ylabel(function_name + '  norm = ' + str('norm' in kwargs))
        plt.xlabel('time after start of experimental chain [min]')

        fig.savefig(graph_dir() + path.sep + 'AsymmetricH_' + self.trajectories[0].size + '_' + self.originalMovies[
            0] + '_' + function_name + '.pdf',
                    format='pdf', pad_inches=1, bbox_inches='tight')


def series_plot(function, *args, **kwargs):
    for name in firstMovies():
        print(name)
        k = Series(name)

        # Ma = k.Max(function)
        # print(Ma)

        # Mi = k.Min(function)
        # print(Mi)

        valueList = [function(x, *args, **kwargs) for x in k.trajectories]
        if len(k) > 0:
            k.plot(valueList, function.__name__, *args, norm=Maze(k.trajectories[0]).exit_size, **kwargs)


# def series_stat_test(function, *args, **kwargs):
#     return
#     for name in firstMovies():
#         print(name)
#         k = Series(name)
#
#         valueList = [function(x, *args, **kwargs) for x in k.trajectories]
#
#
#         Delta = [abs(a - b) for a, b in zip(valueList[:-1], valueList[1:])]
#
#     return


# series_stat_test(AttemptNumber)

""" Help to load the shape """


def Loader():
    firstMovies = [x[:-4] for x in listdir(SaverDirectories['ant'] + path.sep + 'AssymetricH_Series')]

    for firstMovie in firstMovies:
        solver = 'ant'
        lines = read_text_file(TextFileDirectory, firstMovie + '.txt')

        fps = int(lines[1].split(' ')[2][:2])

        shapes = []
        shapesPerMovie = [lines[index].split(' ')[2:] for index in range(1, len(lines)) if
                          lines[index].split(' ')[0].startswith('S')]
        for i in range(len(shapesPerMovie)):
            shapes = shapes + [sh.replace('\n', '') + 'ASH' for sh in shapesPerMovie[i] if sh.replace('\n', '') != '']

        dictionary = Zipper(lines)
        trajectories, sizes = [], []

        for originalMovie in [x.replace('S', '') for x in dictionary.keys()]:
            for root, dirs, files in walk(trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + 'Asymmetric H'):
                for file in files:
                    if originalMovie in file and file.endswith('.mat'):
                        trajectories.append(file)
                        sizes.append(root.split(path.sep)[-2])

        if len(trajectories) != len(shapes) or len(trajectories) != len(sizes):
            breakpoint()
        i = -1
        for (tr, size, shape) in zip(trajectories, sizes, shapes):
            i += 1
            if not (NewFileName(tr, size, shape, 'exp') in listdir(SaverDirectories['ant']) + listdir(
                    SaverDirectories['ant'] + path.sep + 'OnceConnected')) and not (
                    tr in read_text_file(SaverDirectories['ant'], 'DontLoad.txt')):
                print(tr)

                if 'failed' in NewFileName(tr, size, shape, 'exp'):
                    winner = False
                else:
                    winner = True

                x = Load_Experiment(solver, tr, [], winner, 0, 0, 0, fps, False,
                                    shape=shape, size=size)

                Complaints = x.Inspect()[1]
                for com in Complaints:
                    print(com)

                x.time = dictionary['S' + x.filename.split('_')[2]]

                index = int(input('index     '))
                y = x.play(1, indices=[index, index + 3])
                print(str([y.x_error[0], y.y_error[0], y.angle_error[0]]))
                breakpoint()
                x = Load_Experiment(solver, tr, [], winner, y.x_error[0], y.y_error[0], y.angle_error[0],
                                    fps,
                                    False, shape=shape, size=size)

                x.play(5)
                breakpoint()
                x.save()
