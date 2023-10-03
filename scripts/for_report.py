import cv2
import os
from albumentations import (Compose, Normalize, RandomBrightnessContrast,
                            RandomCrop, Resize, RGBShift, ShiftScaleRotate,
                            SmallestMaxSize, MotionBlur, GaussianBlur,
                            MedianBlur, Blur, RandomRotate90, HorizontalFlip,
                            VerticalFlip, HueSaturationValue, RandomSizedCrop,
                            IAASharpen)
import argparse
import face_alignment
import numpy as np
img_transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.25),
        ShiftScaleRotate(rotate_limit=[-15,15], shift_limit=[0.15,0.15], scale_limit=[0.9, 1.1]),
])

# src = "data_verify/report/src"
# dst = "data_verify/report/dst"

# for file in os.listdir(src):
#     file_path = os.path.join(src,file)
#     image = cv2.imread(file_path)
    
#     t_image = img_transform(image=image)["image"]

#     dst_path = os.path.join(dst,file)
#     cv2.imwrite(dst_path,t_image)





frame = "data_verify/report/landmark-process/frame"
face_crop = "data_verify/report/landmark-process/face_crop"
face_landmark = "data_verify/report/landmark-process/face-landmark"
landmark = "data_verify/report/landmark-process/landmark"
dilate = "data_verify/report/landmark-process/dilate"
blur = "data_verify/report/landmark-process/blur"
result = "data_verify/report/landmark-process/result"



for file in os.listdir(frame):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    image = cv2.imread(os.path.join(frame,file))
    fl_image = cv2.imread(os.path.join(frame,file))

    preds = fa.get_landmarks(image)
    if preds == None or len(preds) == 0:
        print(file+ "no have face")
        

    listY = [int(y) for x,y in preds[0]]
    face_size = max(listY) - min(listY)

    thickness = int(face_size / 10)
    blur_size = int(face_size / 10)

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
    dilate_img = np.zeros(image.shape, dtype=np.uint8)

    for key, value in pred_types.items():
        cur_landmarks = preds[0][value].tolist()

        if key in ["lips", "eye1", "eye2"]:
            cur_landmarks.append(cur_landmarks[0])
        for i in range(len(cur_landmarks)-1):
            pt1 = (int(cur_landmarks[i][0]), int(cur_landmarks[i][1]))
            pt2 = (int(cur_landmarks[i+1][0]), int(cur_landmarks[i+1][1]))

            cv2.line(fl_image, pt1, pt2, (0, 0, 255), 2)
            cv2.line(landmark_vis, pt1, pt2, (255, 255, 255), 2)
            cv2.line(dilate_img, pt1, pt2, (255, 255, 255), thickness)
            



    blurred_img = cv2.blur(dilate_img, (blur_size, blur_size))

    scaled_image = blurred_img / 255

    result_image = image * scaled_image

    non_zero_pixels = np.nonzero(result_image)

    min_y = np.min(non_zero_pixels[0])
    max_y = np.max(non_zero_pixels[0])
    min_x = np.min(non_zero_pixels[1])
    max_x = np.max(non_zero_pixels[1])

    face_crop_image = image[min_y:max_y+1, min_x:max_x+1]
    fl_image = fl_image[min_y:max_y+1, min_x:max_x+1]
    landmark_vis = landmark_vis[min_y:max_y+1, min_x:max_x+1]
    dilate_img = dilate_img[min_y:max_y+1, min_x:max_x+1]
    blurred_img = blurred_img[min_y:max_y+1, min_x:max_x+1]
    result_image = result_image[min_y:max_y+1, min_x:max_x+1]

    cv2.imwrite(os.path.join(face_crop, file), face_crop_image)
    cv2.imwrite(os.path.join(face_landmark, file), fl_image)
    cv2.imwrite(os.path.join(landmark, file), landmark_vis)
    cv2.imwrite(os.path.join(dilate, file), dilate_img)
    cv2.imwrite(os.path.join(blur, file), blurred_img)
    cv2.imwrite(os.path.join(result, file), result_image)
