from __future__ import print_function
import numpy as np
import time
import cv2
from os import path
import sys
import imutils

video_directory = path.join('C:\\', 'Users', 'tabea', 'Desktop')

# movie_name = 'large_20210726224634_20210726225802_Full.avi'
movie_name = 'small2_20201220091633_20201220091921_Full.avi'

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


if __name__ == '__main__':
    height = 500
    speed_up = 10
    fps = 50

    frame_shape = get_frame(cap, height=height).shape
    (h, w) = tuple(np.array(frame_shape)[:2])
    final_output_shape = (h, int(frame_shape[1]), 3)
    video_writer = cv2.VideoWriter(path.join(video_directory, sys.argv[0].split('/')[-1].split('.')[0] +
                                             movie_name[:-4] + '.mp4v'),
                                   cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                   (final_output_shape[1], final_output_shape[0]))

    count = 0
    while True:
        time.sleep(1 / fps)
        frame = get_frame(cap, height=height)

        if cv2.waitKey(1) & 0xFF == ord('q') or type(frame) is bool:
            break

        img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2BGRA)
        cv2.imshow('frame', img)
        video_writer.write(img)


        count += speed_up
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
