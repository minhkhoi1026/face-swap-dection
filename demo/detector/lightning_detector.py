import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import abc
import pandas as pd

import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.opt import Config
from src.model import MODEL_REGISTRY
from src.augmentation import TRANSFORM_REGISTRY
from demo.dataset import DATASET_REGISTRY
from src.dataset.demo_dataset import DemoDataset
from demo.detector.base_detector import BaseDetector
from demo.extractor.frame_extractor import FrameExtractor
from demo.extractor.face_fafi_extractor import FaceFAFIExtractor

class TorchLightningDetector(BaseDetector):
    def __init__(self,
                 name,
                 cfg_path,
                 frame_width=1280,
                 frame_height=720,
                 thickness_percentage=10, 
                 blur_percentage=10,
                ):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.face_extractor = FaceFAFIExtractor(thickness_percentage, blur_percentage)
        self.frame_extractor = FrameExtractor(frame_width, frame_height)
        
        self.cfg = Config(cfg_path)
        self.model = self.create_detector_model(self.cfg)
        self.grad_cam_model, self.cam = self.create_gradcam_model(self.cfg)
        
    def create_detector_model(self, cfg):
        model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
        return model.load_from_checkpoint(cfg['global']['resume'],
                                        cfg=cfg,
                                        strict=True)
    
    @abc.abstractmethod
    def create_gradcam_model(self, cfg):
        return None, None
    
    def get_grad_cam(self, batch):
        return []
    
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
        
        dataset = DATASET_REGISTRY.get(cfg["dataset"]["name"])(
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
                all_grad_cam.extend(self.get_grad_cam(batch))
                
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
    
    def predict(self, video_bytes: bytes, sampling: int=10):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()
            
            frame_infos = self.extract_data(temp_file.name, sampling)
            result = self._eval(frame_infos)
            return pd.DataFrame(result)

    
    