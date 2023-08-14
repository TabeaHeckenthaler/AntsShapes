from __future__ import print_function
import numpy as np
import time
import cv2
from Directories import trackedAntMovieDirectory, video_directory
from os import path
import sys
import imutils



def get_frame(c: cv2.VideoCapture, height: int = None):
    if c.isOpened():
        ret, frame = c.read()
        if ret is True:
            frame = imutils.resize(frame, height=height)
            return frame
        else:
            return False


def find_maximum_pixels(frames) -> (int, int):
    return tuple(np.max(np.array(frames), axis=0)[:2])


def merge_frames(frames: list, final_output_shape: (int, int, int), shift: np.array, division=False):
    """

    :param frames: list of numpy arrays, rgb of frames
    :param final_output_shape: (height, width, 3) of the final image
    :return: numpy array with shape final_output_shape
    """
    output = np.ones(final_output_shape, dtype="uint8") * 255
    for i, frame in enumerate(frames):
        if type(frame) is not bool:  # frame can also be false, then we dont display anything
            if division:
                frame[:, -5:, :] = 0
                frame[:, 0:5, :] = 0
            output[shift[i][0]:
                   shift[i][0] + frame.shape[0],
                   shift[i][1]:
                   shift[i][1] + frame.shape[1]] = frame

    return output


if __name__ == '__main__':
    movie_names = ['XLH_4100022_1_ants.avi',
                   'XLH_4100023_1_ants.avi',
                   'XLH_4100026_1_ants.avi'
                   ]

    addresses = [path.join(trackedAntMovieDirectory, 'Slitted', 'H', 'XL', 'Output Videos', movie_name) for movie_name
                 in
                 movie_names]
    cap_list = [cv2.VideoCapture(address) for address in addresses]
    height = 500
    frames0 = [get_frame(cap, height=height).shape for cap in cap_list]
    (h, w) = find_maximum_pixels(frames0)
    fps = 50

    final_output_shape = (h, int(np.sum(frames0, axis=0)[1]), 3)
    shift = np.transpose(np.cumsum([np.zeros(len(frames0), dtype=int), [0, *np.array(frames0)[:, 1][:-1]]], axis=1))
    video_writer = cv2.VideoWriter(video_directory + sys.argv[0].split('/')[-1].split('.')[0] + '.mp4v',
                                   cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                   (final_output_shape[1], final_output_shape[0]))

    count = 0
    while True:
        time.sleep(1 / fps)
        frames = [get_frame(cap, height=height) for cap in cap_list]
        img = cv2.cvtColor(np.array(merge_frames(frames, final_output_shape, shift, division=True)), cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', img)
        video_writer.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q') or len([frame for frame in frames if type(frames) is bool]) == len(frames0):
            break

        count += 5
        [cap.set(cv2.CAP_PROP_POS_FRAMES, count) for cap in cap_list]

    [cap.release() for cap in cap_list]
    cv2.destroyAllWindows()
    video_writer.release()
