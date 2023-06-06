import cv2
from src.model import MODEL_REGISTRY
import numpy as np
import random
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from albumentations import Compose, Normalize
import cv2
# https://github.com/albumentations-team/albumentations/issues/1246
cv2.setNumThreads(0)
from albumentations.pytorch.transforms import ToTensorV2

from src.utils.loading import load_image_label
from src.utils.retinex import automatedMSRCR
# from . import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
class FaceSpoofingDistillationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_path: str,
        split_file: str,
        model_config: dict,
        oversampling: bool,
        num_classes: int=2,
        img_transform=None,
        img_normalize=None,
    ):
        """
        Constructor for face spoofing training dataset, will be passed to data generator

        Args:
            source_path (str): path to dataset directory which contain two sub-directories named fake and real
            split_file (str): path to split file for train/val/test
            model_config (dict): model configuration dictionary, usually loaded from config file
            oversampling (str): whether to oversampling the dataset so that number of fake and real samples are equal
            num_classes (int, optional): format of output label (1 is label encoding and 2 is one hot encoding). Defaults to 2.
            img_transform (Callable, optional): image augmentation transform. Defaults to None.
            img_normalize (Callable, optional): image normalizer, execute at the end of transform.
        """
        
        self.image_paths, self.labels = load_image_label(source_path, split_file, oversampling)
        self.img_transform = img_transform
        self.img_normalize = img_normalize
        self.num_classes = num_classes
        model = MODEL_REGISTRY.get(model_config["model"]["name"])(model_config)
        self.teacher_model = model.load_from_checkpoint(model_config["model"]['weight_path'],
                                       cfg=model_config,
                                       strict=True)
        self.teacher_model.eval()
        self.temperature = model_config["model"]["temperature"]

    def __len__(self):
        return len(self.image_paths)
    
    def __preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_transform:
            img = self.img_transform(image=img)["image"] # only works with albumentations
        
        img = self.img_normalize(image=img)["image"] # only works with albumentations
        
        return img
        

    def __getitem__(self, idx):
        return {
            "img": self.__preprocess_image(self.image_paths[idx]),
            "label": self.labels[idx],
            "img_path": self.image_paths[idx]
        }

    def collate_fn(self, batch):
        imgs = torch.stack([x["img"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        if self.num_classes == 2:
            labels = F.one_hot(labels, self.num_classes)
        with torch.no_grad():
            logits = self.teacher_model({"imgs": imgs})["logits"] / self.temperature
            teacher_labels = nn.Softmax(dim=1)(logits)
        img_paths =  [x["img_path"] for x in batch]
            
        batch_as_dict = {
            "imgs": imgs,
            "hard_labels": labels,
            "teacher_labels": teacher_labels,
            "img_paths": img_paths
        }

        return batch_as_dict

if __name__ == "__main__":
    from src.utils.opt import Opts
    from albumentations import (Compose, Normalize, RandomBrightnessContrast,
                            RandomCrop, Resize, RGBShift, ShiftScaleRotate,
                            SmallestMaxSize, MotionBlur, GaussianBlur,
                            MedianBlur, Blur, RandomRotate90, HorizontalFlip,
                            VerticalFlip, HueSaturationValue, RandomSizedCrop,
                            IAASharpen)
    img_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=[-10,10], shift_limit=[0.15,0.15], scale_limit=[0.75, 1.25]),
        RandomBrightnessContrast(brightness_limit=0.25),
    ])
    img_normalize = Compose([
        Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ToTensorV2()])
    dataset = FaceSpoofingDistillationDataset(
        source_path="fsd-deepfakerapp/deepfaker_app/face_crop/test",
        split_file="val_split.csv",
        model_config=Opts(cfg="configs/distillation.yml").parse_args().teacher_model,
        oversampling=True, 
        img_transform=img_transform,
        img_normalize=img_normalize,
    )
    batch = dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(1)])
    print(batch["imgs"].shape)
    print(batch["teacher_labels"].shape)
    print(batch["hard_labels"].shape)
