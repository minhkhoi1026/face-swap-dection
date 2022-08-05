"""source: https://github.com/ondyari/FaceForensics + https://github.com/ipazc/mtcnn"""
import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN
import tensorflow as tf
import argparse

detector = MTCNN()

def extract_faces(image, output_path, prefix):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with tf.device('/GPU:0'):
        faces = detector.detect_faces(img)
    crop_face = 0
    for id, face in enumerate(faces):
        x, y, w, h = face['box']
        crop_face = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_path, '{}_{:02d}.png'.format(prefix, id)), crop_face)

def extract_frames(data_path, output_path, prefix_images, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv."""
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        prefix_face_img = '{}_{:04d}'.format(prefix_images, frame_num)
        extract_faces(image, output_path, prefix_face_img) # extract faces from single image
        frame_num += 1
    reader.release()
    
def extract_individual(individual_path, output_path, individual_name):
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
                       image_path, prefix)

def extract_all_individual(data_path, output_path):
    for individual in tqdm(os.listdir(data_path)):
        extract_individual(os.path.join(data_path, individual), output_path, individual)



tf.debugging.set_log_device_placement(True)

extract_all_individual(os.path.join("dataset", "casia_fasd", "train_release"), "train")
extract_all_individual(os.path.join("dataset", "casia_fasd", "test_release"), "test")
