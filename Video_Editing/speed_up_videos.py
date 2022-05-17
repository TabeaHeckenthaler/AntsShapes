from __future__ import print_function
import numpy as np
import time
import cv2
from os import path
import sys
import imutils
from tqdm import tqdm

# video_directory = path.join('C:\\', 'Users', 'tabea', 'Desktop')
video_directory = '{0}{1}phys-guru-cs{2}ants{3}honeypot{4}honeypot{5}Udi{6}Ants_videos{7}' \
                  'Paratrechina longicornis{8}2016{9}2016-09-19 (ILAN (fixed release point test (large)))'.\
    format(*[path.sep for _ in range(10)])

if not path.exists(video_directory):
    raise ValueError('Your video directory does not exist')

# movie_name = 'large_20210726224634_20210726225802_Full.avi'
movie_name = 'S2660011_(4k(30fps)_larger ring).MP4'

address = path.join(video_directory, movie_name)
cap = cv2.VideoCapture(address)


def get_frame(c: cv2.VideoCapture, height: int = None):
    if c.isOpened():
        ret, frame = c.read()
        if ret is True:
            frame = imutils.resize(frame, height=height)
            return frame
        else:
            return False


def open_video_writer():
    frame_shape = get_frame(cap, height=height).shape
    (h, w) = tuple(np.array(frame_shape)[:2])
    final_output_shape = (h, int(frame_shape[1]), 3)
    saved_video = path.join(video_directory, sys.argv[0].split('/')[-1].split('.')[0] +
                                             movie_name[:-4] + '.mp4v')
    print('Movie saved in ', saved_video)
    video_writer = cv2.VideoWriter(saved_video, cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                   (final_output_shape[1], final_output_shape[0]))
    return video_writer


if __name__ == '__main__':
    frames = [5838, 11742]
    height = 800
    speed_up = 20
    fps = 50
    video_writer = open_video_writer()

    for count in tqdm(range(frames[0], frames[1], speed_up)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        time.sleep(1 / fps)
        frame = get_frame(cap, height=height)

        if cv2.waitKey(1) & 0xFF == ord('q') or type(frame) is bool:
            break

        img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2BGRA)
        # cv2.imshow('frame', img)
        video_writer.write(img)

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
