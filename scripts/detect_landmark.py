"""source: https://github.com/1adrianb/face-alignment """
import os
import cv2
from tqdm import tqdm
import argparse
import face_alignment
import json
from numba import jit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="path of source data", required=True)
    parser.add_argument("--dest", help="path of destination frame store", required=True)
    
    return parser.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
def extract_frames(data_path, output_path, prefix):
    image = cv2.imread(data_path)
    preds = fa.get_landmarks(image)
    lists = preds[0].tolist()
    
    os.makedirs(output_path, exist_ok = True)
    with open('{}.json'.format(os.path.join(output_path,prefix)), 'w', encoding ='utf8') as json_file:
        json.dump(lists, json_file, separators=(',',':'))


@jit
def detect_all_image(source_path, dest_path):
    """Extracts all videos file structure"""
    for path, _, files in os.walk(source_path):
        relative_path = os.path.relpath(path, source_path)
        files.sort()
        for img_file in tqdm(files, desc=relative_path):
            # prefix of image file name
            img_name = os.path.splitext(img_file)[0]
            
            # folder for store image base on type of image
            image_path = os.path.join(dest_path, relative_path)
            
            extract_frames(os.path.join(source_path, relative_path, img_file),
                        image_path, img_name)

args = parse_args()

source_path = args.source
dest_path = args.dest
detect_all_image(source_path, dest_path)
