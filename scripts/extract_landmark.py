""" https://github.com/1adrianb/face-alignment """
import os
import cv2
from tqdm import tqdm
import argparse
import face_alignment
import numpy as np
from numba import jit



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="path of source data", required=True)
    parser.add_argument("--dest", help="path of destination frame store", required=True)
    parser.add_argument("--thickness", help="percentage of thickness of the line and width of image", type=int, default=10)
    parser.add_argument("--blur", help="percentage of kernel to blur and width of image", type=int, default=10)
    parser.add_argument("--crop", help="crop the image (default: False)", const=True, default=False)
    
    return parser.parse_args()


@jit
def extract_landmark(image_path, output_path, prefix, thickness_percentage, blur_percentage, is_crop):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    image = cv2.imread(image_path)
    preds = fa.get_landmarks(image)
    if preds == None or len(preds) == 0:
        print(image_path+ "no have face")
        return

    _, iW, _ = image.shape

    thickness = int(iW * thickness_percentage / 100)
    blur = int(iW * blur_percentage / 100)

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

    if is_crop:
        non_zero_pixels = np.nonzero(result_image)

        min_y = np.min(non_zero_pixels[0])
        max_y = np.max(non_zero_pixels[0])
        min_x = np.min(non_zero_pixels[1])
        max_x = np.max(non_zero_pixels[1])

        result_image = result_image[min_y:max_y+1, min_x:max_x+1]


    os.makedirs(output_path, exist_ok = True)
    cv2.imwrite("{}_landmark.png".format(os.path.join(output_path, prefix)), result_image)


@jit
def extract_all_image(source_path, dest_path, thickness, blur, crop):
    for path, _, files in os.walk(source_path):
        relative_path = os.path.relpath(path, source_path)
        files.sort()
        for file in tqdm(files, desc=relative_path):
            image_name = os.path.splitext(file)[0]
            input_path = os.path.join(source_path, relative_path, file)

            output_path = os.path.join(dest_path, relative_path)
            
            extract_landmark(input_path, output_path, image_name, thickness, blur, crop)

args = parse_args()

source_path = args.source
dest_path = args.dest
thickness = args.thickness
blur = args.blur
crop = args.crop
extract_all_image(source_path, dest_path, thickness, blur, crop)