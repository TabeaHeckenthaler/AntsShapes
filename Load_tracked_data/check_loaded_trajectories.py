from trajectory_inheritance.trajectory import get
import os


def delete_from_file(deletable_line):
    with open("check_trajectories.txt", "r") as input_:
        with open("temp.txt", "w") as output_:
            for line in input_:
                if line.strip("\n").split(':')[0] != deletable_line:
                    output_.write(line)
    os.replace('temp.txt', 'check_trajectories.txt')


if __name__ == '__main__':
    file1 = open('check_trajectories.txt', 'r')
    trajectories_filenames = [line.split(':')[0].replace('\n', '')
                              for line in file1.readlines() if len(line) > 5]
    file1.close()

    start = 18
    for i, trajectories_filename in enumerate(trajectories_filenames[start:], start=start):
        print(trajectories_filename)
        print(i)
        x = get(trajectories_filename)
        x.play(step=1)

        if bool(int(input('OK? '))):
            delete_from_file(trajectories_filename)

# TODO: There are a few videos in check_trajectories.txt which are not correctly loaded.
# TODO: Fix the easy image processing mistakes.
# TODO: Recalculate connections with the new configuration space.
# TODO: Not closest node is the end space, but choose the one behind the wall... is this the case for all trajectories?
