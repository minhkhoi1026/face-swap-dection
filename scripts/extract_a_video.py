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
import pandas as pd

tf.get_logger().setLevel('ERROR')

from functools import wraps
import sys
import io



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path of input data (video or image)", required=True)
    
    return parser.parse_args()



fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
@jit
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
def detect_face_by_mtcnn(image):
    with tf.device('/GPU:0'):
        faces = detector.detect_faces(image)
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
@jit
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

@jit
def extract_faces(image, face_output, landmark_output):
    # print(type(image))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    iH, iW, _ = img.shape

    min_x1, min_y1, max_x1, max_y1 = detect_face_by_mtcnn(img)
    min_x2, min_y2, max_x2, max_y2 = detect_face_by_face_mesh(img)
    # print(name, (min_x1, min_y1, max_x1, max_y1), (min_x2, min_y2, max_x2, max_y2))

    min_x = max(min(min_x1, min_x2), 0)
    min_y = max(min(min_y1, min_y2), 0)
    max_x = min(max(max_x1, max_x2), iW-1)
    max_y = min(max(max_y1, max_y2), iH-1)
    if max_x <= min_x or max_y <= min_y:
        print("no face in {}".format(name))
        return False

    while max_x - min_x > face_size:
        min_x //= 2
        min_y //= 2
        max_x = math.ceil(max_x / 2)
        max_y = math.ceil(max_y / 2)
        iH //= 2
        iW //= 2

    if max_x <= min_x or max_y <= min_y:
        print("detect face fail in {}".format(name))
        return False
    
    # print(min_x, min_y, max_x, max_y)
    resized_img = cv2.resize(image, (iW,iH))


    landmark = extract_landmark(resized_img)
    if landmark is None:
        print("detect landmark fail in {}".format(name))
        return False
    cv2.imwrite(landmark_output, landmark)

    crop_face = resized_img[min_y:max_y+1, min_x:max_x+1]
    cv2.imwrite(face_output, crop_face)
    return True

def extract_image(data_path):
    dest_path = os.path.splitext(data_path)
    os.makedirs(os.path.join(dest_path,"face"), exist_ok = True)
    os.makedirs(os.path.join(dest_path,"landmark"), exist_ok = True)

    landmark_output = os.path.join(dest_path, "landmark.png")
    face_output = os.path.join(dest_path, "face.png")

    rows = []
    if extract_faces(image, face_output, landmark_output):
        rows.append(face_output, landmark_output, 1)
    df = pd.DataFrame(rows, columns=["filepath", "variant", "label"])
    df.to_csv(os.path.join(dest_path, "all.csv"), index=None)


def extract_video(data_path):
    """Method to extract frames, either with ffmpeg or opencv."""
    dest_path = os.path.splitext(data_path)
    os.makedirs(os.path.join(dest_path,"face"), exist_ok = True)
    os.makedirs(os.path.join(dest_path,"landmark"), exist_ok = True)

    reader = cv2.VideoCapture(data_path)
    frame_num = -1
    nframe = int(1 / sampling_ratio) # choose 1 frame per 1/x frames

    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for frame_num in tqdm(range(total_frames)):
        success, image = reader.read()
        
        # only process success frame
        if not success:
            break
        
        # uniform sampling
        if not frame_num % nframe == 0:
            continue
        
        name = '{:06d}.png'.format(frame_num)
        landmark_output = os.path.join(dest_path, "landmark", name)
        face_output = os.path.join(dest_path, "face", name)
        # extract faces from frame
        if extract_faces(image, face_output, landmark_output): # extract faces from single image
            rows.append(face_output, landmark_output, 1)
    reader.release()
    df = pd.DataFrame(rows, columns=["filepath", "variant", "label"])
    df.to_csv(os.path.join(dest_path, "all.csv"), index=None)

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



args = parse_args()

source_path = args.input
sampling_ratio = 1
confidence_threshold = 0.9
face_size = 200
thickness_percentage = 10
blur_percentage = 10



if is_image(source_path):
    extract_image(source_path)
elif is_video(source_path):
    extract_video(source_path)
else:
    print("Please choose a image or video!")
