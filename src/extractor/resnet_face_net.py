# Code apdated by RainbowDango from https://github.com/tomas-gajarsky/facetorch/blob/main/conf/analyzer/predictor/embed/r50_vggface_1m.yaml
from math import e
from src.extractor import EXTRACTOR_REGISTRY
import torch
from src.extractor.base_extractor import ExtractorNetwork
import os
import gdown
from loguru import logger
import torch
import timm

BASE_RESNET_NAME = "resnet50"

@EXTRACTOR_REGISTRY.register()
class ResnetFaceExtractor(ExtractorNetwork):
    def __init__(self, version, in_channels=3, freeze=False, weight_local_dir="~/.fsd/extractor/resnetface/"):
        super().__init__()
        weight_local_dir = os.path.expanduser(weight_local_dir)
        
        self.version_to_drive_id = {
            "resnet50_vgg_face": "1v4rU9uXtHI2IeHW7pE2scVDDPm8xADBB",
            "resnet50_vgg_face_1m": "1I4jQ0XnKOz9sOri0neJGMpb37ntGL5eA",
            "resnet50_flickr_face": "1UM5dEiR16J7E5Cjbwm4OyRVw6IUVZ6m4",
        } 
        self.extractor = self.get_model(version, weight_local_dir, in_channels)
        self.feature_dim = self.extractor.num_features  # num_features for consistency with other models
        if freeze:
            self.freeze()

    def get_model(self, version, weight_local_dir, in_channels):
        # model = timm.create_model(BASE_RESNET_NAME, in_chans=in_channels, pretrained=False)
        
        if version in self.version_to_drive_id.keys():
            model = timm.create_model(BASE_RESNET_NAME, in_chans=in_channels, pretrained=False)
            model_weights = self.get_weights(version, weight_local_dir)
            model.load_state_dict(model_weights, strict=False)
        elif version == "resnet50":
            model = timm.create_model(BASE_RESNET_NAME, in_chans=in_channels, pretrained=True)
        else:
            raise ValueError(f"Unknown version {version}")
        
        return model
    
    def get_weights(self, version, weight_local_dir):
        weight_local_path = os.path.join(weight_local_dir, f"{version}.pth")
        
        if not os.path.exists(weight_local_path):
            drive_id = self.version_to_drive_id[version]
            logger.debug(f"Downloading weights for VGGFaceExtractor to {weight_local_path}")
            os.makedirs(weight_local_dir, exist_ok=True)
            url = f"https://drive.google.com/uc?&id={drive_id}&confirm=t"
            gdown.download(url, output=weight_local_path, quiet=False)
        
        # https://github.com/1adrianb/unsupervised-face-representation
        init_weights = torch.load(weight_local_path, map_location=torch.device('cpu'))['state_dict']
        return {k.replace('module.base_net.', ''):v for k, v in init_weights.items()}
            
    def forward(self, x):
        x = self.extractor.forward_features(x)
        # https://github.com/huggingface/pytorch-image-models/blob/8ff45e41f7a6aba4d5fdadee7dc3b7f2733df045/timm/models/resnet.py#L724C9-L724C32
        x = self.extractor.global_pool(x)
        return x

if __name__ == "__main__":
    model = ResnetFaceExtractor("resnet50_vgg_face_1m")
    print(model(torch.randn(4, 3, 224, 224)).shape)
