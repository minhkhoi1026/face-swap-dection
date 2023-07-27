import cv2
import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, ReplayCompose
import cv2
# https://github.com/albumentations-team/albumentations/issues/1246
cv2.setNumThreads(0)
from albumentations.pytorch.transforms import ToTensorV2

from src.utils.loading import load_image_variant_label
from src.utils.retinex import automatedMSRCR
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FaceSpoofingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list,
        image_variant_paths: list,
        num_classes: int=2,
        img_transform=None,
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
        
        self.image_paths = image_paths
        self.image_variant_paths = image_variant_paths
        self.img_transform = img_transform
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

    def __getitem__(self, idx):
        img, img_variant = self.__preprocess_image(self.image_paths[idx])

        return {
            "img": img,
            "img_variant": img_variant,
            "label": 1,
        }

    def collate_fn(self, batch):
        labels = torch.tensor([x["label"] for x in batch])
        if self.num_classes == 2:
            labels = F.one_hot(labels, self.num_classes)
            
        batch_as_dict = {
            "imgs": torch.stack([x["img"] for x in batch]),
            "img_variants": torch.stack([x["img_variant"] for x in batch]),
            "labels": labels
        }

        return batch_as_dict
