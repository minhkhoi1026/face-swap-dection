from albumentations import (ReplayCompose, RandomBrightnessContrast,
                            Resize, ShiftScaleRotate, RandomRotate90,
                            HorizontalFlip, VerticalFlip)
import cv2
# https://github.com/albumentations-team/albumentations/issues/1246
cv2.setNumThreads(0)

from . import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
def train_classify_replay_tf(img_size: int):
    return ReplayCompose([
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.25),
        ShiftScaleRotate(rotate_limit=[-15,15], shift_limit=[0.15,0.15], scale_limit=[0.75, 1.25]),
    ])
