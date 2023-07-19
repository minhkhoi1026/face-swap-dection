import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import face_alignment
from functools import partial

import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms

from src.utils.opt import Config
from src.model import MODEL_REGISTRY
from src.augmentation import TRANSFORM_REGISTRY
from src.dataset.demo_dataset import DemoDataset
from demo.detector.base_detector import BaseDetector
from demo.extractor.frame_extractor import FrameExtractor
from demo.extractor.face_fafi_extractor import FaceFAFIExtractor
from demo.utils.gradcam import show_cam_on_image

from pytorch_grad_cam import GradCAMPlusPlus

def create_detector_model(cfg):
    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    return model.load_from_checkpoint(cfg['global']['resume'],
                                        cfg=cfg,
                                        strict=True)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class GradCAMCompatibleModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        # (B, 2*C, H, W)
        batch_dict = {
            "imgs": batch[:, :3, :, :],
            "img_variants": batch[:, 3:, :, :]
        }
        result = self.model(batch_dict)["logits"]
        return result

class RoopDetector(BaseDetector):
    def __init__(self, cfg_path: str="configs/double_head_resnet_infer.yml",
                 frame_width=1280,
                 frame_height=720,
                 thickness_percentage=10, 
                 blur_percentage=10,
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.face_extractor = FaceFAFIExtractor(thickness_percentage, blur_percentage)
        self.frame_extractor = FrameExtractor(frame_width, frame_height)
        
        self.cfg = Config(cfg_path)
        self.model = create_detector_model(self.cfg)
        
        self.grad_cam_model = GradCAMCompatibleModel(create_detector_model(self.cfg))
        self.cam = GradCAMPlusPlus(model=self.grad_cam_model,
                            target_layers=[
                                self.grad_cam_model.model.img_extractor.extractor.layer4[-1],
                                self.grad_cam_model.model.img_variant_extractor.extractor.layer4[-1]
                                # self.grad_cam_model.model.img_extractor.extractor.blocks[-3],
                                # self.grad_cam_model.model.img_variant_extractor.extractor.blocks[-1],
                                ],
                            use_cuda=torch.cuda.is_available(),
                            # reshape_transform=partial(reshape_transform, height=14, width=14)
                            )
        self.cam.batch_size = self.cfg['data_loader']['test']['args']['batch_size']
        
    def create_detector_model(self, cfg):
        model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
        return model.load_from_checkpoint(cfg['global']['resume'],
                                        cfg=cfg,
                                        strict=True)
    
    def _eval(self, frame_infos):
        cfg = self.cfg
        self.model.eval().to(self.device)
        
        image_size = cfg["model"]["input_size"]
        image_transform_test = TRANSFORM_REGISTRY.get(cfg["dataset"]["transform"]["test"])(
            img_size=image_size)
        img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()
        
        face_paths = []
        fafi_paths = []
        for frame_info in frame_infos:
            for data in frame_info["predict"]:
                face, fafi = data["face_path"], data["fafi_path"]
                face_paths.append(face)
                fafi_paths.append(fafi)
        
        dataset = DemoDataset(
            img_transform=image_transform_test,
            img_normalize=img_normalize,
            image_paths=face_paths,
            image_variant_paths=fafi_paths,
        )

        data_loader = DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg["data_loader"]["test"]["args"],
        )

        data_iter = iter(data_loader)

        all_logits = []
        all_grad_cam = []
        while True:
            try:
                batch = next(data_iter)

                for key in batch:
                    if key in ["imgs","img_variants"]:
                        batch[key] = batch[key].to(self.device)
                
                logits = self.model(batch)['logits']
                logits = F.softmax(logits)[:, 1]
                
                all_logits.extend(logits.tolist())
                all_grad_cam.extend(self.cam(input_tensor=torch.cat((batch["imgs"], batch["img_variants"]), dim=1),
                                            targets=None,
                                            eigen_smooth=False,
                                            aug_smooth=False))
                
            except StopIteration:
                break

        idx = 0
        for frame in frame_infos:
            for data in frame["predict"]:
                data["grad_cam"] = all_grad_cam[idx]
                data["score"] = all_logits[idx] * 100 # to percentage
                idx += 1

        return frame_infos
    
    def extract_data(self, video_path: str, sampling):
        frame_infos = self.frame_extractor.extract(video_path, sampling)
        for frame_info in frame_infos:
            frame_info["predict"] = self.face_extractor.extract(frame_info["frame_path"])
        return frame_infos
    
    def predict(self, video_bytes: bytes, sampling: int=100):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()
            
            frame_infos = self.extract_data(temp_file.name, sampling)
            result = self._eval(frame_infos)
            return pd.DataFrame(result)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    detector = RoopDetector()
    x = detector.predict(open("data_verify/200_ntthau.mp4", "rb").read())
    
    image_path= x.iloc[0]["frame_path"]
    bbox = x.iloc[0]["predict"][0]["bbox"]
    grayscale_cam = x.iloc[0]["predict"][0]["grad_cam"]
    print(grayscale_cam.shape)
    rgb_img = cv2.imread(image_path, 1)
    rgb_img = np.float32(rgb_img) / 255
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, bbox)
    cv2.imwrite(f'detector_cam.jpg', cam_image)

    
    