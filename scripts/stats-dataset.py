import os
import cv2
import shutil
import pandas as pd


video_source_dir = "dataset_v1/videos"

val_id = ['video023', 'video104', 'video009', 'video008', 'video022', 'video103', 'video102', 'video007', 'video105', '073', '068', '200', '034', '066', '071', '070', '069', '202', '067', '032', '074', '201', '072', '030', '204', '203', '065', '031', '033']
test_id = ['video033', 'video032', 'video034', 'video127', 'video128', 'video131', 'video031', 'video129', 'video126', '084', '044', '009', '043', '041', '055', '040', '059', '080', '007', '057', '005', '081', '083', '006', '082', '008', '056', '058', '042']

def get_id_video(path):
    path = os.path.basename(path)
    return path.split('.')[0].split('_')[0]

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

# for label in ["fake","real"]:
#     d = {'train':0,'val':0,'test':0}

#     for dts in ["roop","deepfaker"]:
#         path = os.path.join(video_source_dir,dts,label)
#         for file in os.listdir(path):
#             if 'mp4' not in file:
#                 print(file)
#             videoname = get_id_video(file)
#             subset = 'val' if videoname in val_id else 'test' if videoname in test_id else 'train'
#             d[subset] += frame_count(os.path.join(path,file))

#     print(label, d)


metadata_path = "dataset_v1/imgs/{}.csv"

for subset in ["train","val","test"]:
    # Read the CSV file
    data = pd.read_csv(metadata_path.format(subset))
    real = len([l for l in data["label"] if l == 0])
    fake = len([l for l in data["label"] if l == 1])
    print("real",real)
    print("fake",fake)
    print("all",real+fake)

