import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
from os import path
from DataFrame.Altered_DataFrame import find_directory_of_original
from DataFrame.import_excel_dfs import df_ant_excluded
from Directories import network_dir, home
import numpy as np
from trajectory_inheritance.get import get
from tqdm import tqdm


def extend_time_series_to_match_frames(ts, frame_len):
    indices_to_ts_to_frames = np.cumsum([1 / (int(frame_len / len(ts) * 10) / 10)
                                         for _ in range(frame_len)]).astype(int)
    ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
    return ts_extended


def find_random_frames():
    from DataFrame.import_excel_dfs import df_ant_excluded
    copy_ = df_ant_excluded.copy()
    df_ant_excluded = find_directory_of_original(copy_)
    df_ant_excluded = df_ant_excluded[df_ant_excluded['size'] == 'S (> 1)']

    with open(path.join(network_dir, 'time_series_selected_states.json'), 'r') as json_file:
        time_series_dict = json.load(json_file)
        json_file.close()

    ts = {}
    choosen_frames = {}
    dir_frames = {}

    for i in range(len(df_ant_excluded)):
        filename = df_ant_excluded['filename'].iloc[i]
        x = get(filename)
        ts[filename] = extend_time_series_to_match_frames(time_series_dict[filename], len(x.frames))

        # find indices of cg in ts[filename]
        if 'cg' in ts[filename]:
            dir_frames[filename] = []
            inds = np.where(np.array(ts[filename]) == 'cg')[0] + x.frames[0]
            choosen_frames[filename] = np.random.choice(inds, 3, replace=False)
            ns = choosen_frames[filename] // 67948
            for n, ind in zip(ns, inds):
                d = ('\\').join(df_ant_excluded['directory'].iloc[i].split('\\')[:-1] + eval(
                    df_ant_excluded['VideoChain'].iloc[i])) + '.MP4'
                f = ind - n * 67948
                dir_frames[filename].append((d, int(f)))

        else:
            choosen_frames[filename] = []

    # map chosen frames to df_ant_excluded
    df_ant_excluded['frames'] = df_ant_excluded['filename'].map(choosen_frames)

    # save dir_frames to json
    with open('dir_frames.json', 'w') as json_file:
        json.dump(dir_frames, json_file, indent=4)
        json_file.close()


find_random_frames()

# read dir_frames.json
with open('dir_frames.json', 'r') as json_file:
    dir_frames = json.load(json_file)
    json_file.close()


# Function to handle mouse clicks on the image
def click_event(event, x, y, flags, param):
    global click_counts
    if event == cv2.EVENT_LBUTTONDOWN:
        click_counts[param] = click_counts.get(param, 0) + 1
        # print(f"Click count for frame {param}: {click_counts[param]}")


# Initialize click counts dictionary
click_counts = {}


# Process each video file
for i in range(2):
# for i in range(len(df_ant_excluded)):
    row = df_ant_excluded.iloc[i]
    # Open the video file
    movies_frames = dir_frames[row['filename']]
    for movie, frame in movies_frames:
        video = cv2.VideoCapture(movie)

        # Get total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # turn string of list without , to list
        # frames = [int(i) for i in row['frames'].replace('[', '').replace(']', '').split()]

        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)

        # Read the frame
        ret, frame = video.read()

        # Check if the frame was read successfully
        if ret:
            # Generate the output image file name
            image_file = f"{output_directory}{row['filename']}_{frame}.jpg"

            # Save the frame as an image
            cv2.imwrite(image_file, frame)

            # Display the image and wait for a click
            cv2.imshow("Frame", frame)
            cv2.setMouseCallback("Frame", click_event, param=f"{row['filename']}_{frame}")
            cv2.waitKey(0)
        else:
            print(f"Error reading frame {frame} from video {row['filename']}")

    # Release the video file
    video.release()

# Save the click counts to a JSON file
with open("click_counts_outside.json", "w") as json_file:
    json.dump(click_counts, json_file, indent=4)

# Close all OpenCV windows
cv2.destroyAllWindows()
