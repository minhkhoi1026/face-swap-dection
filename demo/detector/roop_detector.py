import streamlit as st
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
import cv2
from tqdm import tqdm
import argparse
import math
from numba import jit
import numpy as np
import pandas as pd
import face_alignment
from functools import wraps, partial
import sys
import io
import re
from pytorch_grad_cam import ScoreCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

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

# @st.cache_resource
def create_face_alignment_model():
    return  face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# @st.cache_resource
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

def get_patch_size_from_name(name):
    regex_pattern = r".*?patch([^_]+)_.*"
    match = re.match(regex_pattern, name)
    return int(match.group(1))

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
    def __init__(self, cfg_path: str="configs/for_demo.yml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fa = create_face_alignment_model()
        self.pred_types = {'face': slice(0, 17),
                    'eyebrow1': slice(17, 22),
                    'eyebrow2': slice(22, 27),
                    'nose': slice(27, 31),
                    'nostril': slice(31, 36),
                    'eye1': slice(36, 42),
                    'eye2': slice(42, 48),
                    'lips': slice(48, 60),
                    'teeth': slice(60, 68),
                    }
        self.thickness_percentage = 10
        self.blur_percentage = 10
        self.frame_width = 1280
        self.frame_height = 720
        self.cfg = Config(cfg_path)
        self.model = create_detector_model(self.cfg)
        patch_size = 14
        self.grad_cam_model = GradCAMCompatibleModel(self.model)
        self.cam = GradCAMPlusPlus(model=self.grad_cam_model,
                            target_layers=[
                                self.grad_cam_model.model.img_extractor.extractor.blocks[-1].norm1, 
                                self.grad_cam_model.model.img_variant_extractor.extractor.blocks[-1].mlp,
                                # self.grad_cam_model.model.feat_attention,   
                                ],
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=partial(reshape_transform, height=patch_size, width=patch_size))
        self.cam.batch_size = self.cfg['data_loader']['test']['args']['batch_size']
         
    def extract_face(self, image, dest, frame_id):
        frame_path = os.path.join(dest, "{:06d}.png".format(frame_id))
        cv2.imwrite(frame_path, image)

        landmark_pred = self.fa.get_landmarks(image)
        
        if not landmark_pred:
            return {"frame_path":frame_path, "frame_id": frame_id, "predict": []}
        
        face_landmark = []

        for id, face in enumerate(landmark_pred):
            landmark_vis = np.zeros(image.shape, dtype=np.uint8)

            listY = [int(y) for x,y in face]
            face_size = max(listY) - min(listY)

            thickness = int(face_size * self.thickness_percentage / 100)
            blur = int(face_size * self.blur_percentage / 100)

            for key, value in self.pred_types.items():
                cur_landmarks = face[value].tolist()

                if key in ["lips", "eye1", "eye2"]:
                    cur_landmarks.append(cur_landmarks[0])
                for i in range(len(cur_landmarks)-1):
                    pt1 = (int(cur_landmarks[i][0]), int(cur_landmarks[i][1]))
                    pt2 = (int(cur_landmarks[i+1][0]), int(cur_landmarks[i+1][1]))

                    cv2.line(landmark_vis, pt1, pt2, (255, 255, 255), thickness)
            blurred_img = cv2.blur(landmark_vis, (blur, blur))
            scaled_image = blurred_img / 255
            result_image = image * scaled_image

            non_zero_pixels = np.nonzero(result_image)
            min_y = np.min(non_zero_pixels[0])
            max_y = np.max(non_zero_pixels[0])
            min_x = np.min(non_zero_pixels[1])
            max_x = np.max(non_zero_pixels[1])

            landmark_crop = result_image[min_y:max_y+1, min_x:max_x+1]
            face_crop = image[min_y:max_y+1, min_x:max_x+1]

            landmark_path = os.path.join(dest, "{:06d}_{:02d}_landmark.png".format(frame_id, id))
            face_path = os.path.join(dest, "{:06d}_{:02d}_face.png".format(frame_id, id))
            
            cv2.imwrite(landmark_path, landmark_crop)
            cv2.imwrite(face_path, face_crop)

            face_landmark.append({"face_landmark":(face_path, landmark_path),"bbox":(min_x,min_y,max_x,max_y)})

        return {"frame_path":frame_path, "frame_id": frame_id, "predict": face_landmark}
    
    def _eval(self, data_dir, frame_data):
        cfg = self.cfg
        self.model.eval().to(self.device)
        
        image_size = cfg["model"]["input_size"]
        image_transform_test = TRANSFORM_REGISTRY.get(cfg["dataset"]["transform"]["test"])(
            img_size=image_size)
        img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()
        
        face_paths = []
        landmark_paths = []
        for frame in frame_data:
            for data in frame["predict"]:
                face, landmark = data["face_landmark"]
                face_paths.append(face)
                landmark_paths.append(landmark)
        
        dataset = DemoDataset(
            img_transform=image_transform_test,
            img_normalize=img_normalize,
            image_paths=face_paths,
            image_variant_paths=landmark_paths,
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
        for frame in frame_data:
            for data in frame["predict"]:
                data["grad_cam"] = all_grad_cam[idx]
                data["score"] = all_logits[idx] * 100 # to percentage
                idx += 1

        return frame_data
    
    def extract_frames(self, data_path, sampling = 1):
        """Method to extract frames, either with ffmpeg or opencv."""
        dest_path = tempfile.mkdtemp()

        face_landmark = []

        reader = cv2.VideoCapture(data_path)

        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Calculate the aspect ratio
        aspect_ratio = width / height
        
        new_width = self.frame_width
        new_height = self.frame_height

        if aspect_ratio > new_width / new_height:
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = int(new_height * aspect_ratio)
        if width > new_width:
            print(f"This video will be downscaled from ({width},{height}) to ({new_width},{new_height}).")

        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in tqdm(range(total_frames)):
            success = reader.grab()

            if frame_id % sampling:
                continue
            
            success, frame = reader.retrieve()
            
            if not success:
                continue

            if width > new_width:
                frame = cv2.resize(frame, (new_width, new_height))

            # extract faces from frame
            face_landmark.append(self.extract_face(frame, dest_path, frame_id)) # extract faces from single image

        reader.release()

        return dest_path, face_landmark      
    
    # @st.cache_data
    def predict(_self, video_bytes: bytes, sampling: int=100):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()
            
            frame_dir, face_landmark = _self.extract_frames(temp_file.name, sampling)
            result = _self._eval(frame_dir, face_landmark)
            return pd.DataFrame(result)

if __name__ == "__main__":
    detector = RoopDetector()
    x = detector.predict(open("data_verify/200_ntthau.mp4", "rb").read())
    
    image_path= x.iloc[1]["predict"][0]["face_landmark"][0]
    grayscale_cam = x.iloc[1]["predict"][0]["grad_cam"]
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'detector_cam.jpg', cam_image)
    