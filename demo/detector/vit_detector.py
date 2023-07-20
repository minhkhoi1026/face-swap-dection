import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import cv2
import numpy as np
from functools import partial

import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.utils.opt import Config
from src.model import MODEL_REGISTRY
from demo.extractor.frame_extractor import FrameExtractor
from demo.extractor.face_fafi_extractor import FaceFAFIExtractor
from demo.utils.gradcam import show_cam_on_image, GradCAMCompatibleModel, reshape_transform
from demo.detector.lightning_detector import TorchLightningDetector

class ViTDetector(TorchLightningDetector):
    def __init__(self, 
                 name,
                 cfg_path: str="configs/double_head_vit_infer.yml",
                 frame_width=1280,
                 frame_height=720,
                 thickness_percentage=10, 
                 blur_percentage=10,
                ):
        super().__init__(name, cfg_path, frame_width, frame_height, thickness_percentage, blur_percentage)

    def create_gradcam_model(self, cfg):
        grad_cam_model = GradCAMCompatibleModel(self.create_detector_model(cfg))
        cam = GradCAMPlusPlus(model=grad_cam_model,
                            target_layers=[
                                grad_cam_model.model.img_extractor.extractor.blocks[-8],
                                grad_cam_model.model.img_variant_extractor.extractor.blocks[-8],
                                ],
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=partial(reshape_transform, height=14, width=14)
                            )
        cam.batch_size = cfg['data_loader']['test']['args']['batch_size']
        return grad_cam_model, cam
    
    def get_grad_cam(self, batch):
        return self.cam(input_tensor=torch.cat((batch["imgs"], batch["img_variants"]), dim=1),
                                            targets=None,
                                            eigen_smooth=False,
                                            aug_smooth=False)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    detector = ViTDetector()
    x = detector.predict(open("data/all_videos/fake/010_txdang.mp4", "rb").read())
    
    image_path= x.iloc[0]["frame_path"]
    bbox = x.iloc[0]["predict"][0]["bbox"]
    grayscale_cam = x.iloc[0]["predict"][0]["grad_cam"]
    print(grayscale_cam.shape)
    rgb_img = cv2.imread(image_path, 1)
    rgb_img = np.float32(rgb_img) / 255
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, bbox)
    cv2.imwrite(f'detector_cam_vit.jpg', cam_image)

    