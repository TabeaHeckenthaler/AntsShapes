import cv2
import os

video_directory = os.path.join('P:\\', 'Tabea', 'Human Experiments',
                               'Raw Data and Videos', '2022-04-11', 'Videos', 'Small near maze')

movie_names = os.listdir(video_directory)
# movie_names = ['NVR_ch2_main_20220411112203_20220411112223.asf',
#                'NVR_ch2_main_20220411112223_20220411112226.asf',
#                'NVR_ch2_main_20220411112226_20220411112347.asf',
#                'NVR_ch2_main_20220411112347_20220411112400.asf',
#                'NVR_ch2_main_20220411112400_20220411112458.asf',
#                'NVR_ch2_main_20220411112458_20220411112503.asf',
#                'NVR_ch2_main_20220411112503_20220411112536.asf',
#                'NVR_ch2_main_20220411112536_20220411112537.asf',
#                'NVR_ch2_main_20220411112537_20220411112721.asf']

frames_dict = {}
for movie_name in movie_names:
    address = os.path.join(video_directory, movie_name)
    cap = cv2.VideoCapture(address)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_dict[movie_name] = frames

print(frames_dict)
