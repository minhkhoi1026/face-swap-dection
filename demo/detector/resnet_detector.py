import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import cv2
import numpy as np

import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.utils.opt import Config
from src.model import MODEL_REGISTRY
from demo.extractor.frame_extractor import FrameExtractor
from demo.extractor.face_fafi_extractor import FaceFAFIExtractor
from demo.utils.gradcam import GradCAMCompatibleModel
from demo.detector.lightning_detector import TorchLightningDetector

class ResNetDetector(TorchLightningDetector):
    def __init__(self,
                 name, 
                 cfg_path: str="configs/inference/double_head_resnet_fafi_hybrid.yml",
                 frame_width=1280,
                 frame_height=720,
                 thickness_percentage=10, 
                 blur_percentage=10,
                 *args,
                 **kwargs
                ):
        super().__init__(name,
                         cfg_path, 
                         frame_width, 
                         frame_height, 
                         thickness_percentage, 
                         blur_percentage, 
                         *args,
                         **kwargs
                         )

    def create_gradcam_model(self, cfg):
        grad_cam_model = GradCAMCompatibleModel(self.create_detector_model(cfg))
        cam = GradCAMPlusPlus(model=grad_cam_model,
                            target_layers=[
                                grad_cam_model.model.img_extractor.extractor.layer4[-1],
                                grad_cam_model.model.img_variant_extractor.extractor.layer4[-1]
                                ],
                            use_cuda=torch.cuda.is_available(),
                            )
        cam.batch_size = cfg['data_loader']['test']['args']['batch_size']
        return grad_cam_model, cam
    
    def get_grad_cam(self, batch):
        n = batch["imgs"].shape[0]
        targets = [ClassifierOutputTarget(1) for i in range(n)]
        return self.cam(input_tensor=torch.cat((batch["imgs"], batch["img_variants"]), dim=1),
                                            targets=targets,
                                            eigen_smooth=False,
                                            aug_smooth=False)
    

if __name__ == "__main__":
    import argparse
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', choices=['msr', 'fafi'])
    feature = parser.parse_args().feature

    cfg = f"configs/inference/double_head_resnet_{feature}_hybrid.yml"
    detector = ResNetDetector("resnet",cfg)
    x = detector.predict(open("data_verify/200_ntthau.mp4", "rb").read())
    
    image_path= x.iloc[0]["predict"][0]["face_path"]
    grayscale_cam = x.iloc[0]["predict"][0]["grad_cam"]
    print(grayscale_cam.shape)
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'detector_cam_resnet_{feature}.jpg', cam_image)

    
    