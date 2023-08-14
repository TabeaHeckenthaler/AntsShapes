from matplotlib import pyplot as plt
import pandas as pd
from Directories import home
from back_room_flipping import BackRoomFlipping
import json

# load excel files
date1 = 'SimTrjs_RemoveAntsNearWall=False'
date2 = 'SimTrjs_RemoveAntsNearWall=True'
df_gillespie1 = pd.read_excel(home + '\\Gillespie\\' + date1 + '_sim.xlsx')
df_gillespie2 = pd.read_excel(home + '\\Gillespie\\' + date2 + '_sim.xlsx')
df_ant_excluded = pd.read_excel(home + '\\DataFrame\\final\\df_ant_excluded.xlsx')

direct1 = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\back_room_flipping\\' + \
         date1 + '_sim_flipping.json'
with open(direct1, 'r') as json_file:
    sim_decisions1 = json.load(json_file)

direct2 = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\back_room_flipping\\' + \
         date2 + '_sim_flipping.json'
with open(direct2, 'r') as json_file:
    sim_decisions2 = json.load(json_file)

direct_ant = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\Analysis\\back_room_flipping\\' + \
         'ant_flipping.json'
with open(direct_ant, 'r') as json_file:
    ant_decisions = json.load(json_file)

# merge dataframes
df_ant_flipping_all_sizes = df_ant_excluded[['filename', 'size', 'winner']].copy()
df_ant_flipping_all_sizes['decisions'] = df_ant_flipping_all_sizes['filename'].apply(
    lambda x: ant_decisions[x])

df_gillespie_flipping_all_sizes1 = df_gillespie1[['filename', 'size']].copy()
df_gillespie_flipping_all_sizes1['decisions'] = df_gillespie_flipping_all_sizes1['filename'].apply(
    lambda x: sim_decisions1[x])

df_gillespie_flipping_all_sizes2 = df_gillespie2[['filename', 'size']].copy()
df_gillespie_flipping_all_sizes2['decisions'] = df_gillespie_flipping_all_sizes2['filename'].apply(
    lambda x: sim_decisions2[x])

# only winner
# df_ant_flipping_all_sizes = df_ant_flipping_all_sizes[df_ant_flipping_all_sizes['winner'] == 1]
# in df_ant_flipping_all_sizes change all 'S (> 1)' to 'S'
df_ant_flipping_all_sizes = df_ant_flipping_all_sizes.replace('S (> 1)', 'S')

for size in ['XL', 'L', 'M', 'S'][::-1]:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # for solver, df in zip(['ant', 'sim_no_removal', 'sim_removal'],
    #                       [df_ant_flipping_all_sizes,
    #                       df_gillespie_flipping_all_sizes1,
    #                       df_gillespie_flipping_all_sizes2]):
    for solver, df in zip(['sim_removal', 'ant', ],
                          [ df_gillespie_flipping_all_sizes2, df_ant_flipping_all_sizes,]):
        df_solver_flipping = df[df['size'] == size]
    
        df_solver_flipping['percentage'] = df_solver_flipping['decisions'].apply(
            lambda x: BackRoomFlipping.percentage_backroom_flipped(x))
        df_solver_flipping['total_entrances'] = df_solver_flipping['decisions'].apply(
            lambda x: BackRoomFlipping.total_entrances(x))
    
        min_time = 10
        decisions_reduced = df_solver_flipping['decisions'].apply(lambda x: BackRoomFlipping.reduce_decisions(x, min_time))
        percentage_reduced = decisions_reduced.apply(lambda x: BackRoomFlipping.percentage_backroom_flipped(x))
        total_entrances_reduced = decisions_reduced.apply(lambda x: BackRoomFlipping.total_entrances(x))
        # save reduced decisions in xlsx file
        df_solver_flipping['decisions_reduced'] = decisions_reduced
        df_solver_flipping.to_excel(solver + '_' + size + '_reduced.xlsx')

        axs[0].hist(percentage_reduced,
                    bins=10,
                    alpha=0.5,
                    label=solver + ', N = ' + str(len(df_solver_flipping)),
                    range=(0, 1),
                    density=True)
        axs[0].axvline(percentage_reduced.mean(), color='k', linestyle='dashed', linewidth=1)
        axs[0].text(percentage_reduced.mean() + 0.01, 0.5, solver, rotation=90, verticalalignment='center')
        axs[0].legend(loc='upper right')
        axs[0].set_ylim([0, 4.5])
        axs[0].set_xlabel('percentage reentrances to slit flipped')
        axs[0].set_ylabel('frequency')
    
        # plot histogram
        n, bins, bar_container = axs[1].hist(total_entrances_reduced,
                                             bins=25,
                                             alpha=0.5,
                                             label=solver + ', N = ' + str(len(df_solver_flipping)),
                                             range=(0, 50),
                                             density=True)
        axs[1].legend(loc='upper right')

        # draw vertical line at bins where n is maximal
        # axs[1].axvline(bins[n.argmax()], color='k', linestyle='dashed', linewidth=1)
        # axs[1].text(bins[n.argmax()] + 0.01, 0.5, solver, rotation=90, verticalalignment='center')
        axs[1].set_ylim([0, 0.45])
        axs[1].set_xlabel('total reentrances')
        axs[1].set_ylabel('frequency')

    fig.suptitle('size: ' + size + ', min_time: ' + str(min_time) + 's')
    fig.savefig(str(min_time) + 's_' + size + '.png')
    DEBUG = 1

DEBUG = 1
