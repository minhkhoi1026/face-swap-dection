"""source: https://github.com/ondyari/FaceForensics + https://github.com/ipazc/mtcnn + https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md """
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
import cv2
from tqdm import tqdm
# import tensorflow as tf
import argparse
import math
from numba import jit
import numpy as np
import pandas as pd
import face_alignment

# tf.get_logger().setLevel('ERROR')

from functools import wraps
import sys
import io

import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms

from src.utils.opt import Config
from src.model import MODEL_REGISTRY
from src.augmentation import TRANSFORM_REGISTRY
from src.dataset.demo_dataset import DemoDataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path of input data (video or image)", required=True)
    
    return parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
pred_types = {'face': slice(0, 17),
            'eyebrow1': slice(17, 22),
            'eyebrow2': slice(22, 27),
            'nose': slice(27, 31),
            'nostril': slice(31, 36),
            'eye1': slice(36, 42),
            'eye2': slice(42, 48),
            'lips': slice(48, 60),
            'teeth': slice(60, 68),
            }
# @jit
def extract_face(image, dest, frame_id):
    frame_path = os.path.join(dest, "{:06d}.png".format(frame_id))
    cv2.imwrite(frame_path, image)

    landmark_pred = fa.get_landmarks(image)
    
    face_landmark = []

    for id, face in enumerate(landmark_pred):
        landmark_vis = np.zeros(image.shape, dtype=np.uint8)

        listY = [int(y) for x,y in face]
        face_size = max(listY) - min(listY)

        thickness = int(face_size * thickness_percentage / 100)
        blur = int(face_size * blur_percentage / 100)

        for key, value in pred_types.items():
            cur_landmarks = face[value].tolist()

            if key in ["lips", "eye1", "eye2"]:
                cur_landmarks.append(cur_landmarks[0])
            for i in range(len(cur_landmarks)-1):
                pt1 = (int(cur_landmarks[i][0]), int(cur_landmarks[i][1]))
                pt2 = (int(cur_landmarks[i+1][0]), int(cur_landmarks[i+1][1]))

                cv2.line(landmark_vis, pt1, pt2, (255, 255, 255), thickness)
        blurred_img = cv2.blur(landmark_vis, (blur, blur))
        scaled_image = blurred_img / 255
        result_image = image * scaled_image

        non_zero_pixels = np.nonzero(result_image)
        min_y = np.min(non_zero_pixels[0])
        max_y = np.max(non_zero_pixels[0])
        min_x = np.min(non_zero_pixels[1])
        max_x = np.max(non_zero_pixels[1])

        landmark_crop = result_image[min_y:max_y+1, min_x:max_x+1]
        face_crop = image[min_y:max_y+1, min_x:max_x+1]

        landmark_path = os.path.join(dest, "{:06d}_{:02d}_landmark.png".format(frame_id, id))
        face_path = os.path.join(dest, "{:06d}_{:02d}_face.png".format(frame_id, id))
        
        cv2.imwrite(landmark_path, landmark_crop)
        cv2.imwrite(face_path, face_crop)

        face_landmark.append({"face_landmark":(face_path, landmark_path),"bbox":(min_x,min_y,max_x,max_y)})

    return {"frame_path":frame_path, "data":face_landmark}

def extract_image(data_path):
    dest_path = os.path.splitext(data_path)[0]

    landmark_output = os.path.join(dest_path, "landmark.png")
    face_output = os.path.join(dest_path, "face.png")

    rows = []

    image = cv2.imread(data_path)
    if extract_face(image, face_output, landmark_output):
        rows.append((face_output, landmark_output, 1))
    df = pd.DataFrame(rows, columns=["filepath", "variant", "label"])
    df.to_csv(os.path.join(dest_path, "all.csv"), index=None)

def test(data_dir, frame_data):
    cfg = Config("configs/for_demo.yml")

    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(cfg['global']['resume'],
                                        cfg=cfg,
                                        strict=True)

    trainer = pl.Trainer(
        gpus=-1
        if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
    )

    model.eval().to(device)
    
    image_size = cfg["model"]["input_size"]
    image_transform_test = TRANSFORM_REGISTRY.get(cfg["dataset"]["transform"]["test"])(
        img_size=image_size)
    img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()
    
    face_paths = []
    landmark_paths = []
    for frame in frame_data:
        for data in frame["data"]:
            face, landmark = data["face_landmark"]
            face_paths.append(face)
            landmark_paths.append(landmark)
    
    dataset = DemoDataset(
        img_transform=image_transform_test,
        img_normalize=img_normalize,
        image_paths=face_paths,
        image_variant_paths=landmark_paths,
    )

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        **cfg["data_loader"]["test"]["args"],
    )

    data_iter = iter(data_loader)

    all_logits = []
    while True:
        try:
            batch = next(data_iter)

            for key in batch:
                if key in ["imgs","img_variants"]:
                    batch[key] = batch[key].to(device)
            
            logits = model(batch)['logits']
            logits = F.softmax(logits)[:, 1]
            
            all_logits.extend(logits.tolist())

        except StopIteration:
            # Đã duyệt qua hết dữ liệu
            break

    idx = 0
    for frame in frame_data:
        for data in frame["data"]:
            data["score"] = all_logits[idx]
            idx += 1

    del trainer
    del cfg
    del model

    return frame_data

def extract_video_from_path(data_path, sampling = 1):
    """Method to extract frames, either with ffmpeg or opencv."""
    dest_path = tempfile.mkdtemp()

    face_landmark = []

    reader = cv2.VideoCapture(data_path)

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Calculate the aspect ratio
    aspect_ratio = width / height
    # Calculate new size
    new_width = 1280
    new_height = 720
    if aspect_ratio > new_width / new_height:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)
    if width > new_width:
        print(f"This video will be downscaled from ({width},{height}) to ({new_width},{new_height}).")

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    # batch_size = 8
    # batch = []

    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(total_frames)):
        # if frame_id == total_frames or len(batch) == batch_size:
        #     batch_np = np.stack(batch)
        #     landmark_pred_batch = fa.get_landmarks_batch(batch_np)
        #     for landmark_pred in landmark_pred_batch:
        #         face_landmark.append(extract_face(batch, landmark_pred, dest_path, frame_id))
        #     batch = []

        success, frame = reader.read()

        if frame_id % sampling:
            continue

        if width > new_width:
            frame = cv2.resize(frame, (new_width, new_height))

        # batch.append(frame)
        # extract faces from frame
        face_landmark.append(extract_face(frame, dest_path, frame_id)) # extract faces from single image

    reader.release()

    # face_landmark = [
    #     {
    #         "frame_path":'data_extract/demo/video/{:06d}.png'.format(id),
    #         "data":[
    #             {
    #                 "face_landmark":(
    #                     'data_extract/demo/video/{:06d}_00_face.png'.format(id),
    #                     'data_extract/demo/video/{:06d}_00_landmark.png'.format(id)
    #                 ),
    #                 "position":(0,0,1,1)
    #             }
    #         ]
    #     }
    #     for id in range(27)]

    return test(dest_path, face_landmark)

def extract_video(video_bytes, sampling = 1):
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(video_bytes)
        temp_file.flush()
        
        return extract_video_from_path(temp_file.name, sampling)


def is_image(file_path):
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return True
        else:
            return False
    except cv2.error:
        return False

def is_video(file_path):
    try:
        video = cv2.VideoCapture(file_path)
        if video.isOpened():
            return True
        else:
            return False
    except cv2.error:
        return False


confidence_threshold = 0.9
face_size = 200
thickness_percentage = 10
blur_percentage = 10



if __name__ == "__main__":
    args = parse_args()
    source_path = args.input

    if is_image(source_path):
        extract_image(source_path)
    elif is_video(source_path):
        import cProfile
        import pstats
        import subprocess
        import time
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()

        extract_video_from_path(source_path, 2)
        extract_video_from_path(source_path, 2)

        elapsed_time = time.time() - start_time
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats('profile_data.prof')
        subprocess.call(['gprof2dot', '-f', 'pstats', 'profile_data.prof', '-o', 'profile_data.dot'])
        # subprocess.call(['./flamegraph.pl', 'profile_data.dot', '-o', 'flamegraph.svg'])

        print(f"The function took {elapsed_time} seconds to execute.")

    else:
        print("Please choose a image or video!")
# a = 0.12/frame
# b = 15
# 
