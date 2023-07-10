import os
import cv2
import shutil


image_dir = "fsd-deepfakerapp/hybrid"
video_source_dir = "data/all_videos"


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

def frame_count(video_path):
    frames = 0
 
    cap = cv2.VideoCapture(video_path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print('Fail!!!')
    cap.release()
    return frames

dfcnt = 0

for subset in ["train","val","test"]:
    imgtype = "face"
    for label in ["fake","real"]:
        path = os.path.join(image_dir,subset,imgtype,label)
        listvideo = set()
        for file in os.listdir(path):
            videoname = file.split("_")[0]
            listvideo.add(videoname)
        dict_list_video[subset+"_"+label] = listvideo

        for videoname in listvideo:
            if "video" in videoname:
                dfcnt += 1

        fcnt = 0
        for file in os.listdir(os.path.join(video_source_dir, label)):
            videoname = file.split("_")[0].split(".")[0]
            if videoname in listvideo:
                fcnt += frame_count(os.path.join(video_source_dir, label, file))

        print(subset,"\t",label,"\t",len(os.listdir(path)),"\t",len(listvideo), '\t', fcnt)
        # listvideo = list(listvideo)
        # listvideo.sort()
        # print(listvideo)

print(dfcnt)

# mistake = [video for video in dict_list_video["train_real"] if video not in dict_list_video["train_fake"]]
# mistake.sort()
# print(len(mistake))
# print(mistake)


# backup = "data_extract/backup/test"


# mistake = [video for video in dict_list_video["test_fake"] if video not in dict_list_video["test_real"]]
# mistake.sort()
# print(len(mistake))
# print(mistake)

# path = os.path.join(image_dir,"test","face","real")
# for file in os.listdir(path):
#     filename = os.path.splitext(file)[0]
#     videoname = filename[:len(filename)-5]
#     if videoname not in [file.split("_")[0] for file in dict_list_video["test_fake"]]:
#         # print(file)
#         # source_path = os.path.join(path, file)
#         # destination_path = os.path.join(backup, file)
#         # shutil.move(source_path, destination_path)
#         pass