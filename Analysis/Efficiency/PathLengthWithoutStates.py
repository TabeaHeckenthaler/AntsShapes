from Analysis.Efficiency.PathLength import *
from Analysis.PathPy.Path import time_series_dict

direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\path_length_without_bStates.json'


class PathLength_without_states(PathLength):
    def __init__(self, x, to_exclude: list):
        super().__init__(x)
        self.name = 'path length without states'
        self.to_exclude = to_exclude

    def calculate_path_length(self, rot: bool = True, frames: list = None, kernel_size=None, max_path_length=np.inf):
        """
        Path length without counting any of the path length within the self.to_exclude states.
        """
        states = PathLength_without_states.make_state_vector(time_series_dict[self.x.filename], len(self.x.frames))

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
            if state not in self.to_exclude:
                d = self.measureDistance(pos1, pos2, ang1, ang2, aver_radius, rot=rot)
                path_length += d
            else:
                test += 1

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


path_length_without_b_dict = PathLength_without_states.get_dict(direct)


if __name__ == '__main__':
    to_exclude = ['b', 'b2', 'b1', 'be']

    # filename = 'L_SPT_4650012_LSpecialT_1_ants (part 1)'
    # x = get(filename)
    # print(PathLength_without_states(x, to_exclude).calculate_path_length())

    # df_chosen = df[df['solver'] == solver]
    df_chosen = myDataFrame.copy()
    df_chosen = df_chosen[df_chosen['maze dimensions'].isin(['MazeDimensions_new2021_SPT_ant.xlsx',
                                                             'MazeDimensions_human.xlsx'])]
    df_chosen = df_chosen[df_chosen['shape'] == 'SPT']
    df_chosen = df_chosen[df_chosen['initial condition'] == 'back']

    # path_length_without_b_dict = PathLength_without_states.create_dict(df_chosen, to_exclude)

    # path_length_without_b_dict = PathLength_without_states.add_to_dict(myDataFrame, direct)
    # PathLength_without_states.save_dict(path_length_without_b_dict, direct)
