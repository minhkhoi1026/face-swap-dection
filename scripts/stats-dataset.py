import os
import cv2
import shutil


image_dir = "fsd-deepfakerapp/hybrid"


def count_total_frames(video_list):
    total_frames = 0

    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        # Retrieve the total number of frames
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

    return total_frames


dict_list_video = dict()

for subset in ["train","val","test"]:
    imgtype = "face"
    for label in ["fake","real"]:
        path = os.path.join(image_dir,subset,imgtype,label)
        listvideo = set()
        for file in os.listdir(path):
            videoname = file.split("_")[0]
            listvideo.add(videoname)
        dict_list_video[subset+"_"+label] = listvideo

        print(subset,"\t",label,"\t",len(os.listdir(path)),"\t",len(listvideo))

mistake = [video for video in dict_list_video["train_real"] if video not in dict_list_video["train_fake"]]
mistake.sort()
print(len(mistake))
print(mistake)


backup = "data_extract/backup/test"


mistake = [video for video in dict_list_video["test_fake"] if video not in dict_list_video["test_real"]]
mistake.sort()
print(len(mistake))
print(mistake)

path = os.path.join(image_dir,"test","face","real")
for file in os.listdir(path):
    filename = os.path.splitext(file)[0]
    videoname = filename[:len(filename)-5]
    if videoname not in [file.split("_")[0] for file in dict_list_video["test_fake"]]:
        # print(file)
        # source_path = os.path.join(path, file)
        # destination_path = os.path.join(backup, file)
        # shutil.move(source_path, destination_path)
        pass