from Analysis.Efficiency.PathLength import *
import os
from Directories import network_dir
from DataFrame.import_excel_dfs import df_ant


class PathLength_without_states(PathLength):
    def __init__(self, x, to_exclude: iter):
        super().__init__(x)
        self.name = 'path length without states'
        self.to_exclude = to_exclude
        self.ts = PathLength_without_states.extend_time_series_to_match_frames(
            time_series_dict[self.x.filename], len(self.x.frames))

    def calculate_path_length(self, rot: bool = True, kernel_size=None):
        """
        Path length without counting any of the path length within the self.to_exclude states.
        """
        position, angle = self.x.position, self.x.angle

        if kernel_size is None:
            kernel_size = 2 * (self.x.fps // 4) + 1
        ps, uaf = self.x.smoothed_pos_angle(position, angle, kernel_size)

        if uaf.size == 0 or ps.size == 0:
            return 0

        aver_radius = self.average_radius()
        path_length = 0
        test = 0

        for pos1, pos2, ang1, ang2, state in \
                zip(ps[:-1], ps[1:], uaf[:-1], uaf[1:], self.ts[:-1]):
            if state not in self.to_exclude:
                d = self.measureDistance(pos1, pos2, ang1, ang2, aver_radius, rot=rot)
                path_length += d
            else:
                test += 1

        print(self.x.filename, path_length)
        if path_length < 0.1:
            print('path length is 0')
        return path_length

    @staticmethod
    def extend_time_series_to_match_frames(ts, frame_len):
        indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                             for _ in range(frame_len)]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended


if __name__ == '__main__':
    with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    # # ######################################### pre_c ##########################################
    # direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\pre_c_pathlength.json'
    # pre_c = {'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}
    # pre_c_pathlength = {}
    #
    # for index, row in list(df_ant.iterrows()):
    #     traj = get(row['filename'])
    #     pre_c_pathlength[traj.filename] = PathLength_without_states(traj, pre_c).calculate_path_length()
    #
    # with open(direct, 'w') as json_file:
    #     json.dump(pre_c_pathlength, json_file)
    #     json_file.close()
    #
    # # ######################################### post_c ##########################################
    # direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\post_c_pathlength.json'
    # post_c_pathlength = {}
    # post_c = {'ab', 'ac', 'b', 'be', 'b1', 'b2'}
    #
    # for index, row in list(df_ant.iterrows()):
    #     traj = get(row['filename'])
    #     post_c_pathlength[traj.filename] = PathLength_without_states(traj, post_c).calculate_path_length()
    #
    # with open(direct, 'w') as json_file:
    #     json.dump(post_c_pathlength, json_file)
    #     json_file.close()

    # direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\all.json'
    # all = {}
    #
    # for index, row in list(df_ant.iterrows())[:10]:
    #     traj = get(row['filename'])
    #     all[traj.filename] = PathLength_without_states(traj, {}).calculate_path_length()
    #
    # DEBUG = 1

    # ######################################### post_c_without_c ##########################################
    direct = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\Efficiency\\post_c_without_c_pathlength.json'
    post_c_without_c = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg'}
    post_c_pathlength_without_c = {}

    for index, row in list(df_ant.iterrows()):
        traj = get(row['filename'])
        post_c_pathlength_without_c[traj.filename] = PathLength_without_states(traj, post_c_without_c).calculate_path_length()

    with open(direct, 'w') as json_file:
        json.dump(post_c_pathlength_without_c, json_file)
        json_file.close()
