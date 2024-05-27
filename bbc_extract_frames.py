import cv2
from os.path import join
import os

video_data_root_path = "/scratch/harshit/BBC/cam1"

video_files = ["take019_30_OS1_0044_4K.MP4", "take019_31_OS1_0044_4K.MP4", "take02_02_OS1_0027_4K.MP4"]
new_folder_paths = ["take019_30/video_frames", "take019_31/video_frames", "take02_02/video_frames"]

for i in range(4):
    video_file = video_files[i]
    new_folder_path = new_folder_paths[i]
    video_path = join(video_data_root_path, video_file)
    cap = cv2.VideoCapture(video_path)

    count = 0

    video_frames_save_path = join(video_data_root_path, new_folder_path)
    if not os.path.exists(video_frames_save_path):
        os.makedirs(video_frames_save_path)

    while True:
        res, frame = cap.read()
        if not res:
            break
        else:
            cv2.imwrite(join(video_frames_save_path, "%06d.jpg" % count), frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    print("Total frames extracted: ", count)
