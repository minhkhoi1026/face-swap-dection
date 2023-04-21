import cv2
import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.loading import load_image_file_paths, generate_label_from_path
from src.utils.retinex import automatedMSRCR


class FaceSpoofingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        oversampling: bool,
        num_class: int=2,
        img_transform=None,
    ):
        """
        Constructor for face spoofing training dataset, will be passed to data generator

        Args:
            data_path (str): path to dataset directory which contain two sub-directories named fake and real
            oversampling (str): whether to oversampling the dataset so that number of fake and real samples are equal
            num_class (int, optional): format of output label (1 is label encoding and 2 is one hot encoding). Defaults to 2.
            pc_transform (_type_, optional): image augmentation transform. Defaults to None.
        """
        self.image_paths = load_image_file_paths(data_path, oversampling)
        self.labels = generate_label_from_path(self.image_paths)
        self.img_transform = img_transform
        self.num_class = num_class

    def __len__(self):
        return len(self.image_paths)
    
    def __preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_transform:
            img = np.asarray(self.img_transform(img))
            
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = np.expand_dims(new_img, -1)
        new_img = automatedMSRCR(new_img, [10, 20, 30])
        new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)
        
        img = img/255.0
        new_img = new_img/255.0
        
        return torch.tensor(img), torch.tensor(new_img)
        

    def __getitem__(self, idx):
        img, msr_img = self.__preprocess_image(self.image_paths[idx])
        
        label = self.labels[idx]

        return {
            "img": img,
            "msr_img": msr_img,
            "label": label,
        }

    def collate_fn(self, batch):
        labels = torch.tensor([x["label"] for x in batch])
        if self.num_class == 2:
            labels = F.one_hot(labels, self.num_class)
            
        batch_as_dict = {
            "imgs": torch.stack([x["img"] for x in batch]),
            "msr_imgs": torch.stack([x["msr_img"] for x in batch]),
            "labels": labels
        }

        return batch_as_dict

if __name__ == "__main__":
    from torchvision import transforms
    dataset = FaceSpoofingDataset(data_path="casia-fasd/sanity", 
                                  oversampling=True, 
                                  img_transform=transforms.Compose(
                                    [ 
                                        transforms.ToPILImage(),
                                        transforms.Resize((128, 128)),
                                        transforms.RandomAffine(degrees=[-10,10], translate=[0.15,0.15], scale=[0.75, 1.25]), 
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ColorJitter(brightness=0.25),
                                    ]
                                    )
                                )
    batch = dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(1)])
    print(batch["imgs"].shape)
    print(batch["msr_imgs"].shape)
    print(batch["labels"].shape)
