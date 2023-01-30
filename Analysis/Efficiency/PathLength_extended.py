from Analysis.Efficiency.PathLength import *
from Analysis.PathPy.Path import time_series_dict
from trajectory_inheritance.get import get


direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\path_length_extended.json'
state_order = ['b2', 'b1', 'be', 'b', 'ab', 'ac', 'cg', 'c', 'eg', 'eb', 'e', 'f', 'h']


class PathLength_extended(PathLength):
    def __init__(self, x):
        super().__init__(x)
        self.name = 'path length extended'
        self.averages = {}

    @classmethod
    def find_averages(cls, df):
        for filename in df['filename']:
            x = get(filename)
            ts = time_series_dict[x.filename]



        DEBUG = 1

    def calculate_path_length(self, rot: bool = True, frames: list = None, kernel_size=None, max_path_length=np.inf):
        """
        Path length extended by average distance to solving
        """
        if self.x.winner:
            return PathLength(self.x).calculate_path_length()

        states = PathLength_extended.make_state_vector(time_series_dict[self.x.filename], len(self.x.frames))

        if frames is None:
            frames = [0, -1]
        position, angle = self.x.position[frames[0]: frames[1]], self.x.angle[frames[0]: frames[1]]

        if kernel_size is None:
            kernel_size = 2 * (self.x.fps // 4) + 1
        ps, uaf = self.x.smoothed_pos_angle(position, angle, kernel_size)

        if uaf.size == 0 or ps.size == 0:
            return 0

        aver_radius = self.average_radius()
        path_length = 0

        test = 0

        for pos1, pos2, ang1, ang2, state in \
                zip(ps[:-1], ps[1:], uaf[:-1], uaf[1:], states[:-1]):
            d = self.measureDistance(pos1, pos2, ang1, ang2, aver_radius, rot=rot)
            path_length += d

            if path_length > max_path_length:
                return path_length

        print(self.x.filename, path_length)
        return path_length

    @staticmethod
    def make_state_vector(states, desired_length) -> list:
        a = np.array(range(desired_length)) / (desired_length / len(states))
        a = a.astype(int)
        a = a.astype(str)

        for index, state in zip(np.unique(a), states):
            a[a == str(index)] = state
        return a.tolist()


# path_length_extended = PathLength_extended.get_dict(direct)

if __name__ == '__main__':
    # filename = 'L_SPT_4650012_LSpecialT_1_ants (part 1)'
    # x = get(filename)
    # print(PathLength_without_states(x, to_exclude).calculate_path_length())

    # df_chosen = df[df['solver'] == solver]
    df_chosen = myDataFrame.copy()
    df_chosen = df_chosen[df_chosen['maze dimensions'].isin(['MazeDimensions_new2021_SPT_ant.xlsx',
                                                             'MazeDimensions_human.xlsx'])]
    df_chosen = df_chosen[df_chosen['shape'] == 'SPT']
    df_chosen = df_chosen[df_chosen['initial condition'] == 'back']

    PathLength_extended.find_averages(df_chosen[df_chosen['winner']])

    path_length_extended = PathLength_extended.create_dict(df_chosen)

    PathLength_extended.save_dict(path_length_extended, direct)
