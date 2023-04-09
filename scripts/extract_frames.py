"""source: https://github.com/ondyari/FaceForensics + https://github.com/ipazc/mtcnn"""
import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN
import tensorflow as tf
import argparse
from numba import jit

tf.get_logger().setLevel('ERROR')


detector = MTCNN()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                        choices = ["casia-fasd", "oulu-npu", "replay-attack"], 
                        help="specify dataset name for frame extraction", 
                        default="casia-fasd")
    parser.add_argument("--source", help="path of source data", required=True)
    parser.add_argument("--dest", help="path of destination frame store", required=True)
    parser.add_argument("--sampling-ratio", help="specify a ratio x for frame sampling (0 < x <= 1)", required=True)
    
    return parser.parse_args()

def extract_faces(image, output_path, prefix):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with tf.device('/GPU:0'):
        faces = detector.detect_faces(img)
    crop_face = 0
    for id, face in enumerate(faces):
        x, y, w, h = face['box']
        crop_face = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_path, '{}_{:02d}.png'.format(prefix, id)), crop_face)

@jit
def extract_frames(data_path, output_path, prefix_images, sampling_ratio):
    """Method to extract frames, either with ffmpeg or opencv."""
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = -1
    nframe = int(1 / sampling_ratio) # choose 1 frame per 1/x frames
    while reader.isOpened():
        success, image = reader.read()
        
        # only process success frame
        if not success:
            break
        
        # uniform sampling
        frame_num += 1
        if not frame_num % nframe == 0:
            continue
        
        # extract faces from frame
        prefix_face_img = '{}_{:04d}'.format(prefix_images, frame_num)
        extract_faces(image, output_path, prefix_face_img) # extract faces from single image
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
        print(video_name)
        image_path = os.path.join(output_path, image_type)
        
        extract_frames(os.path.join(individual_path, video),
                       image_path, prefix, sampling_ratio)

@jit
def extract_all_individual(source_path, dest_path, sampling_ratio):
    for individual in tqdm(os.listdir(source_path)):
        extract_individual(os.path.join(source_path, individual), dest_path, individual, sampling_ratio)

args = parse_args()

source_path = args.source
dest_path = args.dest
sampling_ratio = float(args.sampling_ratio)
extract_all_individual(source_path, dest_path, sampling_ratio)
