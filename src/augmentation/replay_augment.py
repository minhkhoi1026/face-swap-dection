from albumentations import (ReplayCompose, RandomBrightnessContrast,
                            Resize, ShiftScaleRotate, HorizontalFlip,
                            VerticalFlip)
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
        ShiftScaleRotate(rotate_limit=[-10,10], shift_limit=[0.15,0.15], scale_limit=[0.75, 1.25]),
        RandomBrightnessContrast(brightness_limit=0.25),
    ])
