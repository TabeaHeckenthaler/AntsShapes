import json
from DataFrame.dataFrame import myDataFrame


def initialize_dictionary():
    #
    # with open('need_to_be_retracked.txt', 'w') as file:
    #     json.dump(need_to_be_retracked, file)
    # need_to_be_retracked = {name: True for name in df['filename'].sort_values()}

    with open('retracked.txt', 'r') as file:
        retracked_list = json.load(file)
    have_been_retracked = {name: name in retracked_list for name in df['filename'].sort_values()}
    with open('have_been_retracked.txt', 'w') as file:
        json.dump(have_been_retracked, file)


if __name__ == '__main__':
    df = myDataFrame[(myDataFrame['solver'] == 'ant') &
                     (myDataFrame['initial condition'] == 'back') &
                     (myDataFrame['maze dimensions'] == 'MazeDimensions_new2021_SPT_ant.xlsx')]
    initialize_dictionary()

    with open('need_to_be_retracked.txt', 'r') as file:
        need_to_be_retracked = json.load(file)

    with open('have_been_retracked.txt', 'r') as file:
        have_been_retracked = json.load(file)

    df['need_to_be_retracked'] = df['filename'].map(need_to_be_retracked)
    df['have_been_retracked'] = df['filename'].map(have_been_retracked)

    print('DONE: ', (df['have_been_retracked']).sum())
    print('TO GO: ', (df['need_to_be_retracked'].sum() - df['have_been_retracked'].sum()))


    def video(filename):
        return filename.split('_')[2]
    retrack_df = df[df['need_to_be_retracked'] & ~df['have_been_retracked']]
    retrack_df['video_filename'] = df['filename'].apply(video)
    retrack_df.sort_values('video_filename', inplace=True)
    retrack_df.index = [i for i in range(1, len(retrack_df)+1)]

    DEBUG = 1
