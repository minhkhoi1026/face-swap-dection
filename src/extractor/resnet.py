import timm
from . import EXTRACTOR_REGISTRY
from src.extractor.base_extractor import ExtractorNetwork


@EXTRACTOR_REGISTRY.register()
class ResNetExtractor(ExtractorNetwork):
    def __init__(self, version, in_channels=3, from_pretrained=True, freeze=False):
        super().__init__()
        available_versions = ['mobilenetv3_large_075', 
                              'mobilenetv3_large_100', 
                              'mobilenetv3_large_100_miil', 
                              'mobilenetv3_large_100_miil_in21k', 
                              'mobilenetv3_small_050', 
                              'mobilenetv3_small_075', 
                              'mobilenetv3_small_100']
        assert version in available_versions, f"version must be one of available_versions"
        self.extractor = timm.create_model(version, pretrained=from_pretrained, in_chans=in_channels)
        self.feature_dim = self.extractor.num_features  # num_features for consistency with other models
        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.extractor.forward_features(x)
        x = self.extractor.forward_head(x, pre_logits=True)
        return x
