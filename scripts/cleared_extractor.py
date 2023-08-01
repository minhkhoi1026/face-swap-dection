"""source: https://github.com/ondyari/FaceForensics + https://github.com/ipazc/mtcnn + https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md """
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN
import tensorflow as tf
import argparse
import mediapipe as mp
import math
from numba import jit
import numpy as np
import face_alignment

tf.get_logger().setLevel('ERROR')

from functools import wraps
import sys
import io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="path of source data", required=True)
    parser.add_argument("--dest", help="path of destination image store", required=True)
    parser.add_argument("--sampling-ratio", help="specify a ratio x for frame sampling (0 < x <= 1)", type=float, required=True)
    parser.add_argument("--extract-type", help="choices in {all, frame, face}, default=all", choices=["all", "face", "landmark"], default="all")
    
    return parser.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
def landmark_extract(image):
    landmark_pred = fa.get_landmarks(image)

    if not landmark_pred:
        return None, None

    face_landmark = []

    for id, face in enumerate(landmark_pred):
        landmark_vis = np.zeros(image.shape, dtype=np.uint8)

        listY = [int(y) for x,y in face]
        face_size = max(listY) - min(listY)

        thickness = np.max([int(face_size * thickness_percentage / 100), 1])
        
        blur = np.max([int(face_size * blur_percentage / 100), 1])

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

        return face_crop, landmark_crop
    return None, None

@jit
def extract_faces(image, dest_path, relative_path, prefix):
    height, width, _ = image.shape
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    new_width = 720
    new_height = 1080

    if aspect_ratio > new_width / new_height:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)
    if width > new_width:
        image = cv2.resize(image, (new_width, new_height))

    crop_face, landmark = landmark_extract(image)
    if crop_face is None:
        return
    
    output_path = os.path.join(dest_path, "landmark", relative_path)
    os.makedirs(output_path, exist_ok = True)
    cv2.imwrite(os.path.join(output_path, '{}_landmark.png'.format(prefix)), landmark)
    
    output_path = os.path.join(dest_path, "face", relative_path)
    os.makedirs(output_path, exist_ok = True)
    cv2.imwrite(os.path.join(output_path, '{}.png'.format(prefix)), crop_face)
        

@jit
def extract_frames(data_path, dest_path, relative_path, prefix_images, sampling_ratio):
    """Method to extract frames, either with ffmpeg or opencv."""
    reader = cv2.VideoCapture(data_path)
    frame_num = -1
    nframe = int(1 / sampling_ratio) # choose 1 frame per 1/x frames

    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(total_frames), desc = prefix_images):
        success, image = reader.read()
        
        # only process success frame
        if not success:
            break
        
        # uniform sampling
        if not frame_num % nframe == 0:
            continue
        
        prefix = '{}_{:04d}'.format(prefix_images, frame_num)
        # extract faces from frame
        extract_faces(image, dest_path, relative_path, prefix) # extract faces from single image
    reader.release()

@jit
def extract_all_video(source_path, dest_path, sampling_ratio):
    """Extracts all videos file structure"""
    for path, _, files in os.walk(source_path):
        relative_path = os.path.relpath(path, source_path)
        files.sort()

        print("In folder {}:".format(relative_path))
        # for video in tqdm(files, desc=relative_path):
        for video in files:
            # prefix of image file name
            video_name = os.path.splitext(video)[0]
            
            # folder for store image base on type of image 
            
            extract_frames(os.path.join(source_path, relative_path, video),
                        dest_path, relative_path, video_name, sampling_ratio)

args = parse_args()

source_path = args.source
dest_path = args.dest
sampling_ratio = args.sampling_ratio
extract_type = args.extract_type
confidence_threshold = 0.9
face_size = 200
thickness_percentage = 10
blur_percentage = 10
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
extract_all_video(source_path, dest_path, sampling_ratio)