from DataFrame.dataFrame import myDataFrame
from trajectory_inheritance.trajectory import get
from Directories import df_dir
import json


def correct_a_false_winner(filename, winner=False, comment=''):
    x = get(filename)
    x.winner = winner
    x.save()

    index = myDataFrame[myDataFrame['filename'] == x.filename].index[0]
    myDataFrame_relevant.at[index, 'winner'] = winner
    myDataFrame_relevant.at[index, 'comment'] = comment
    myDataFrame.loc[myDataFrame_relevant.index, :] = myDataFrame_relevant[:]
    myDataFrame.to_json(df_dir)

    with open('winner_dictionary.txt', 'r') as json_file:
        winner_dict = json.load(json_file)
    if x.old_filenames(0) not in winner_dict:
        raise Exception('why?')
    winner_dict[x.old_filenames(0)] = winner
    with open('winner_dictionary.txt', 'w') as json_file:
        json.dump(winner_dict, json_file)


def choose_relevant_experiments(df, shape, solver, winner=None, init_cond='back'):
    """
    Reduce df to relevant experiments
    :param df: dataFrame
    :param shape: shape of the load ('H', 'I', 'SPT'...)
    :param solver: ('human', 'ant', ...)
    :param winner: Do you want to include only successful trajectories?
    :param init_cond: Do you want to restrict the included experiments only to a specific initial condition?
    (front, back or None)
    :return: DataFrame with relevant experiments
    """
    df = df.clone()
    df = df[df['shape'] == shape]
    df = df[df['solver'] == solver]
    if winner is not None:
        df = df[df['winner'] == winner]
    if init_cond == 'back':
        df = df[df['initial condition'] == init_cond]
    return df


if __name__ == '__main__':

    # filename = 'M_SPT_4700022_MSpecialT_1_ants'
    # correct_a_false_winner(filename, winner=True, comment='Extend this trajectory to the end!')

    myDataFrame_relevant = choose_relevant_experiments(myDataFrame, 'SPT', 'ant', winner=True, init_cond='back')
    for i, experiment in myDataFrame_relevant.iterrows():
        print(i)
        x = get(experiment['filename'])
        print(x)
        x.play(frames=[-100, -1], wait=10)

        DEBUG = 1

        # if not bool(int(input('Is winner?'))):
        #     correct_a_false_winner(experiment['filename'])


# TODO: Extend trajectories:
# M_SPT_4700022_MSpecialT_1_ants
# L_SPT_4080033_SpecialT_1_ants (part 1)
# L_SPT_4090010_SpecialT_1_ants (part 1)
