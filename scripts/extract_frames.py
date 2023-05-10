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

tf.get_logger().setLevel('ERROR')

from functools import wraps
import sys
import io



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="path of source data", required=True)
    parser.add_argument("--dest", help="path of destination frame store", required=True)
    parser.add_argument("--sampling-ratio", help="specify a ratio x for frame sampling (0 < x <= 1)", required=True)
    parser.add_argument("--threshold", help="specify a minimum confidence threshold c for face detection (0 < c <= 1)", type=float, default=0.9)
    parser.add_argument("--extract-type", help="choices in {frame, face}", choices=["frame", "face"], default="frame")
    parser.add_argument("--dataset-type", help="choices in {casia, normal}", choices=["casia", "normal"], default="normal")
    parser.add_argument("--face-size", help="maximun of face width, default = 200", type=int, default=200)
    
    return parser.parse_args()


detector = MTCNN(min_face_size=200)

def capture_output(func):
    """Wrapper to capture print output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper

w_detect_face = capture_output(detector.detect_faces)

def detect_face_by_mtcnn(image):
    with tf.device('/GPU:0'):
        faces = w_detect_face(image)
    max_face_size = 0
    iH, iW, _ = image.shape
    min_x = iW
    min_y = iH
    max_x = 0
    max_y = 0
    for face in faces:
        if face["confidence"] >= confidence_threshold:
            x, y, w, h = face['box']
            if w + h > max_face_size:
                max_face_size = w + h
                min_x = x
                min_y = y
                max_x = x + w
                max_y = y + h
    return min_x, min_y, max_x, max_y


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)
def detect_face_by_face_mesh(image):
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(image)

    iH, iW, _ = image.shape
    if not results.multi_face_landmarks:
        return iW, iH, 0, 0
    face_landmarks = results.multi_face_landmarks[0]
    list_x = [int(data_point.x * iW) for data_point in face_landmarks.landmark]
    list_y = [int(data_point.y * iH) for data_point in face_landmarks.landmark]
    return min(list_x), min(list_y), max(list_x), max(list_y)


def extract_faces(image, output_path, prefix):
    # print(type(image))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    iH, iW, _ = img.shape

    min_x1, min_y1, max_x1, max_y1 = detect_face_by_mtcnn(img)
    min_x2, min_y2, max_x2, max_y2 = detect_face_by_face_mesh(img)

    min_x = max(min(min_x1, min_x2), 0)
    min_y = max(min(min_y1, min_y2), 0)
    max_x = min(max(max_x1, max_x2), iW)
    max_y = min(max(max_y1, max_y2), iH)
    if max_x <= min_x or max_y <= min_y:
        # print("no face in {}".format(prefix))
        return

    while max_x - min_x > face_size:
        min_x //= 2
        min_y //= 2
        max_x = math.ceil(max_x / 2)
        max_y = math.ceil(max_y / 2)
        iH //= 2
        iW //= 2

    if max_x <= min_x or max_y <= min_y:
        # print("detect face fail in {}".format(prefix))
        return
    
    resized_img = cv2.resize(image, (iW,iH))

    os.makedirs(output_path, exist_ok = True)
    if extract_type == "frame":
        cv2.imwrite(os.path.join(output_path, '{}.png'.format(prefix)), resized_img)
    else:
        crop_face = resized_img[min_y:max_y+1, min_x:max_x+1]
        cv2.imwrite(os.path.join(output_path, '{}.png'.format(prefix)), crop_face)

@jit
def extract_frames(data_path, output_path, prefix_images, sampling_ratio):
    """Method to extract frames, either with ffmpeg or opencv."""
    os.makedirs(output_path, exist_ok=True)
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
        extract_faces(image, output_path, prefix) # extract faces from single image
    reader.release()

@jit
def extract_individual(individual_path, output_path, individual_name, sampling_ratio):
    """Extracts all videos file structure"""
    for video in os.listdir(individual_path):
        # prefix of image file name
        video_name = os.path.splitext(video)[0]
        prefix = f"{individual_name}_{video_name}"
        
        # folder for store image base on type of image 
        # image have name 1, 2 or HR_1 are real, others are fake
        image_type = "real" if video_name in ["1", "2", "HR_1"] else "fake"
        # print(video_name)
        image_path = os.path.join(output_path, image_type)
        
        extract_frames(os.path.join(individual_path, video),
                       image_path, prefix, sampling_ratio)

@jit
def extract_all_individual(source_path, dest_path, sampling_ratio):
    for individual in tqdm(os.listdir(source_path)):
        extract_individual(os.path.join(source_path, individual), dest_path, individual, sampling_ratio)

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
            # print(video_name)
            image_path = os.path.join(dest_path, relative_path)
            
            extract_frames(os.path.join(source_path, relative_path, video),
                        image_path, video_name, sampling_ratio)

args = parse_args()

source_path = args.source
dest_path = args.dest
confidence_threshold = args.threshold
sampling_ratio = args.sampling_ratio
dataset_type = args.dataset_type
extract_type = args.extract_type
face_size = args.face_size
if dataset_type == "casia":
    extract_all_individual(source_path, dest_path, sampling_ratio)
else:
    extract_all_video(source_path, dest_path, sampling_ratio)