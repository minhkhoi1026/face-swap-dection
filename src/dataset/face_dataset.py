import cv2
import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize
import cv2
# https://github.com/albumentations-team/albumentations/issues/1246
cv2.setNumThreads(0)
from albumentations.pytorch.transforms import ToTensorV2

from src.utils.loading import load_image_label
from src.utils.retinex import automatedMSRCR
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FaceSpoofingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_path: str,
        split_file: str,
        oversampling: bool,
        num_classes: int=2,
        img_transform=None,
        transform_replay=None,
        img_normalize=None
    ):
        """
        Constructor for face spoofing training dataset, will be passed to data generator

        Args:
            source_path (str): path to dataset directory which contain two sub-directories named fake and real
            split_file (str): path to split file for train/val/test
            oversampling (str): whether to oversampling the dataset so that number of fake and real samples are equal
            num_classes (int, optional): format of output label (1 is label encoding and 2 is one hot encoding). Defaults to 2.
            img_transform (Callable, optional): image augmentation transform. Defaults to None.
            img_normalize (Callable, optional): image normalizer, execute at the end of transform.
        """
        
        self.image_paths, self.landmark_paths, self.labels = load_image_label(source_path, split_file, oversampling)
        self.img_transform = img_transform
        self.transform_replay = transform_replay
        self.img_normalize = img_normalize
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)
    
    def __preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_transform:
            img = self.img_transform(image=img)["image"] # only works with albumentations
            
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = np.expand_dims(new_img, -1)
        new_img = automatedMSRCR(new_img, [10, 20, 30])
        new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)
        
        img = self.img_normalize(image=img)["image"] # only works with albumentations
        new_img =  self.img_normalize(image=new_img)["image"] # only works with albumentations
        
        return img, new_img
    
    def __preprocess_image_list(self, image_path_list):
        result = list()
        data = None
        for image_path in image_path_list:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.img_transform and self.transform_replay:
                if data == None:
                    data = self.img_transform(image=img)
                    img = data["image"] # only works with albumentations
                else:
                    img = self.transform_replay(data["replay"], image=img)
                
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            new_img = np.expand_dims(new_img, -1)
            new_img = automatedMSRCR(new_img, [10, 20, 30])
            new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)
            
            img = self.img_normalize(image=img)["image"] # only works with albumentations
            new_img =  self.img_normalize(image=new_img)["image"] # only works with albumentations

            result.append((img, new_img))
        
        return result
        

    def __getitem__(self, idx):
        result = self.__preprocess_image_list((self.image_paths[idx],self.landmark_paths[idx]))
        
        label = self.labels[idx]

        return {
            "img": result[0][0],
            "msr_img": result[0][1],
            "landmark_img": result[1][0],
            "landmark_msr_img": result[1][1],
            "label": label,
            "img_path": self.image_paths[idx]
        }

    def collate_fn(self, batch):
        labels = torch.tensor([x["label"] for x in batch])
        if self.num_classes == 2:
            labels = F.one_hot(labels, self.num_classes)
            
        batch_as_dict = {
            "imgs": torch.stack ([x["img"] for x in batch]),
            "msr_imgs": torch.stack([x["msr_img"] for x in batch]),
            "labels": labels,
            "img_paths": [x["img_path"] for x in batch]
        }

        return batch_as_dict

if __name__ == "__main__":
    from albumentations import (Compose, ReplayCompose, Normalize, RandomBrightnessContrast,
                            RandomCrop, Resize, RGBShift, ShiftScaleRotate,
                            SmallestMaxSize, MotionBlur, GaussianBlur,
                            MedianBlur, Blur, RandomRotate90, HorizontalFlip,
                            VerticalFlip, HueSaturationValue, RandomSizedCrop,
                            IAASharpen)
    img_transform = ReplayCompose([
        Resize(128, 128),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=[-10,10], shift_limit=[0.15,0.15], scale_limit=[0.75, 1.25]),
        RandomBrightnessContrast(brightness_limit=0.25),
    ])
    transform_replay = ReplayCompose.replay
    dataset = FaceSpoofingDataset(
        data_path="casia-fasd/sanity", 
        oversampling=True, 
        img_transform=img_transform,
        transform_replay=transform_replay
    )
    batch = dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(1)])
    print(batch["imgs"].shape)
    print(batch["msr_imgs"].shape)
    print(batch["labels"].shape)
