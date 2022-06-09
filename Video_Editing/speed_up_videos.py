from __future__ import print_function
import numpy as np
import time
import cv2
from os import path
import sys
import imutils
from tqdm import tqdm


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
    video_directory = path.join('C:\\', 'Users', 'tabea', 'Desktop')
    # video_directory = '{0}{1}phys-guru-cs{2}ants{3}honeypot{4}honeypot{5}Udi{6}Ants_videos{7}' \
    #                   'Paratrechina longicornis{8}2016{9}2016-09-19 (ILAN (fixed release point test (large)))'. \
    #     format(*[path.sep for _ in range(10)])

    # video_directory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Videos{5}01_08_2021_(Mexico Building){6}'\
    #     .format(*[path.sep for _ in range(7)])

    if not path.exists(video_directory):
        raise ValueError('Your video directory does not exist')

    # movie_names = {'SSPT_4800001_SSpecialT_1_ants (part 1).avi': [1, 48648],
    #                'SSPT_4800002_SSpecialT_1_ants (part 2).avi': [1, 22682]}

    # movie_names = {'LSPT_4670004_LSpecialT_1_ants (part 1).avi': [1, 49855],
    #                'LSPT_4670005_LSpecialT_1_ants (part 2).avi': [1, 36256]}

    # movie_names = {'S4700010_MSpecialT.MP4': [1045, 49999],
    #                'S4700011_MSpecialT.MP4': [1, 27472]}

    # movie_names = {'S4630011_XLSpecialT.MP4': [1, 40273]}
    movie_names = {'CrazyAnts_Rotating Planet.mp4': [2444, 2648]}

    video_writer = None
    height = 800
    speed_up = 1

    for movie_name, frames in movie_names.items():
        address = path.join(video_directory, movie_name)
        cap = cv2.VideoCapture(address)

        if video_writer is None:
            video_writer = open_video_writer()

        fps = 50
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
