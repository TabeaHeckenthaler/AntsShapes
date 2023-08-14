from Directories import home, network_dir
import pandas as pd
from copy import copy
import json
import numpy as np
import os
from tqdm import tqdm
from trajectory_inheritance.get import get
from matplotlib import pyplot as plt
from colors import colors_humans as colors

df_human_original = pd.read_excel(home + '\\DataFrame\\final\\df_human.xlsx')

with open(os.path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
    time_series_dict = json.load(json_file)
    json_file.close()

vel_threshold = 0.04


class DecisionsTimeHuman:
    def __init__(self, x):
        self.x = copy(x)
        self.ts = self.extend_time_series_to_match_frames(time_series_dict[self.x.filename], len(self.x.frames))
        self.decision_time = None
        self.decision = None
        self.checked_hand_wave = False

    @staticmethod
    def extend_time_series_to_match_frames(ts, frame_len):
        indices_to_ts_to_frames = \
            np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10) for _ in range(frame_len)]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended

    def calc_decision_time(self):
        first_decision = np.where(np.array(self.ts) != 'ab')[0][0]
        self.decision = self.ts[first_decision]

        self.x.smooth(sec_smooth=2)
        vel = self.x.velocity()
        vel_norm = np.linalg.norm(vel, axis=1)
        first_motion = np.where(vel_norm > vel_threshold)[0][0]

        # plt.plot(vel_norm)
        # plt.axvline(first_motion)
        # plt.savefig(self.x.filename + '_vel.png')
        # plt.close()

        # if (first_decision - first_motion) / self.x.fps > 8:
        #     print('first decision is more than 8 seconds after first motion')

        self.decision_time = first_motion / self.x.fps

    @staticmethod
    def correct_start():
        df_human = pd.read_excel('human_decision_time.xlsx')
        for i, row in tqdm(df_human.iterrows(), total=len(df_human)):
            print(i)
            if np.isnan(row['hand_wave']):
                x = get(row['filename'])
                x.open_tracked_video()
                hand_wave = bool(int(input('press 1 if hand wave is detected in the first frame, '
                                           'otherwise press any key')))
                if hand_wave:
                    df_human.loc[i, 'hand_wave'] = 1
                else:
                    df_human.loc[i, 'hand_wave'] = 0
                df_human.to_excel('human_decision_time.xlsx', index=False)

    @staticmethod
    def calc_all_decision_times():
        df_human = pd.read_excel('human_decision_time.xlsx', index_col=0)

        for i, row in tqdm(df_human.iterrows(), total=len(df_human)):
            print(i)
            x = get(row['filename'])
            dth = DecisionsTimeHuman(x)
            dth.calc_decision_time()
            df_human.loc[i, 'decision_time_' + str(vel_threshold)] = dth.decision_time
            df_human.loc[i, 'decision'] = dth.decision
        df_human.to_excel('human_decision_time.xlsx', index=False)

    @staticmethod
    def plot_small():
        df_human = pd.read_excel('human_decision_time.xlsx')

        # choose only the 'Small' size
        df_human = df_human[df_human['hand_wave'] == 1]
        df_human_small = df_human[df_human['size'].isin(['Small Near', 'Small Far'])]

        # group by 'decision'
        decision_groups = df_human_small.groupby('decision')

        # get dataframes of each group
        df_ac = decision_groups.get_group('ac')
        df_b = decision_groups.get_group('b')

        # merge df_ac and df_human_original
        df_ac = df_ac.merge(df_human_original, on='filename')
        df_ac['directory_y'].tolist()

        # find mean and std of decision_time for each group
        mean_ac = df_ac['decision_time_' + str(vel_threshold)].mean()
        std_ac = df_ac['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_ac))
        mean_b = df_b['decision_time_' + str(vel_threshold)].mean()
        std_b = df_b['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_b))

        # plot histogram of decision_time for each group
        plt.figure()
        plt.hist(df_ac['decision_time_' + str(vel_threshold)], alpha=0.5, label='ac')
        plt.hist(df_b['decision_time_' + str(vel_threshold)], alpha=0.5, label='b')
        plt.legend()

        # plot
        plt.figure()
        plt.bar(['ac', 'b'], [mean_ac, mean_b], yerr=[std_ac, std_b])
        plt.ylabel('Decision time [sec]')

        DEBUG = 1

    @staticmethod
    def plot_NC_C():
        df_human = pd.read_excel('human_decision_time.xlsx')

        # choose only the 'Small' size
        df_human = df_human[df_human['hand_wave'] == 1]
        df_C_NC = df_human[~df_human['size'].isin(['Small Near', 'Small Far'])]
        df_C_NC = df_C_NC.merge(df_human_original[['filename', 'average Carrier Number']], on='filename')

        # group by 'decision'
        decision_groups = df_C_NC.groupby(['communication', 'decision'])

        # get dataframes of each group
        df_NC_ac = decision_groups.get_group((0, 'ac'))
        df_NC_b = decision_groups.get_group((0, 'b'))
        df_C_ac = decision_groups.get_group((1, 'ac'))
        df_C_b = decision_groups.get_group((1, 'b'))

        # merge df_ac and df_human_original
        # df_ac = df_ac.merge(df_human_original, on='filename')

        # find mean and std of decision_time for each group
        mean_NC_ac = df_NC_ac['decision_time_' + str(vel_threshold)].mean()
        std_NC_ac = df_NC_ac['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_NC_ac))
        mean_NC_b = df_NC_b['decision_time_' + str(vel_threshold)].mean()
        std_NC_b = df_NC_b['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_NC_b))
        mean_C_ac = df_C_ac['decision_time_' + str(vel_threshold)].mean()
        std_C_ac = df_C_ac['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_C_ac))
        mean_C_b = df_C_b['decision_time_' + str(vel_threshold)].mean()
        std_C_b = df_C_b['decision_time_' + str(vel_threshold)].std()/np.sqrt(len(df_C_b))

        # # plot histogram of decision_time for each group
        # plt.figure()
        # plt.hist(df_NC['decision_time_' + str(vel_threshold)], alpha=0.5, label='ac')
        # plt.hist(df_C['decision_time_' + str(vel_threshold)], alpha=0.5, label='b')
        # plt.legend()

        # plot
        plt.figure()
        plt.bar(['NC_b: ' + str(len(df_NC_b)), 'NC_ac: ' + str(len(df_NC_ac)),
                 'C_b: ' + str(len(df_C_b)), 'C_ac: ' + str(len(df_C_ac)), ],
                [mean_NC_b, mean_NC_ac, mean_C_b, mean_C_ac],
                yerr=[std_NC_b, std_NC_ac, std_C_b, std_C_ac])
        plt.ylabel('Decision time [sec]')

        DEBUG = 1


if __name__ == '__main__':
    # DecisionsTimeHuman.correct_start()
    # DecisionsTimeHuman.calc_all_decision_times()
    # DecisionsTimeHuman.plot_small()
    DecisionsTimeHuman.plot_NC_C()

    DEBUG = 1