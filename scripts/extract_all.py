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



fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
def extract_landmark(image):
    preds = fa.get_landmarks(image)
    if preds == None or len(preds) == 0:
        return None

    listY = [int(y) for x,y in preds[0]]
    face_size = max(listY) - min(listY)

    thickness = int(face_size * thickness_percentage / 100)
    blur = int(face_size * blur_percentage / 100)

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

    landmark_vis = np.zeros(image.shape, dtype=np.uint8)

    for key, value in pred_types.items():
        cur_landmarks = preds[0][value].tolist()

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

    return result_image[min_y:max_y+1, min_x:max_x+1]


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
    min_x = iW-1
    min_y = iH-1
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


def extract_faces(image, dest_path, relative_path, prefix):
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

    if extract_type in ["all","landmark"]:
        landmark = extract_landmark(resized_img)
        if landmark is None:
            return
        output_path = os.path.join(dest_path, "landmark", relative_path)
        os.makedirs(output_path, exist_ok = True)
        cv2.imwrite(os.path.join(output_path, '{}_landmark.png'.format(prefix)), landmark)
    if extract_type in ["all","face"]:
        crop_face = resized_img[min_y:max_y+1, min_x:max_x+1]
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
extract_all_video(source_path, dest_path, sampling_ratio)