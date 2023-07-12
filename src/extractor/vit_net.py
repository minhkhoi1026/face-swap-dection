import timm
from . import EXTRACTOR_REGISTRY
from src.extractor.base_extractor import ExtractorNetwork


@EXTRACTOR_REGISTRY.register()
class VitNetExtractor(ExtractorNetwork):

    def __init__(self, version, in_channels=3, from_pretrained=True, freeze=False):
        super().__init__()
        available_versions = [
            'vit_tiny_patch16_224',
            'vit_tiny_patch16_384',
            'vit_small_patch32_224',
            'vit_small_patch32_384',
            'vit_small_patch16_224',
            'vit_small_patch16_384',
            'vit_base_patch32_224',
            'vit_base_patch32_384',
            'vit_base_patch16_224',
            'vit_base_patch16_384',
            'vit_base_patch8_224',
            'vit_large_patch32_224',
            'vit_large_patch32_384',
            'vit_large_patch16_224',
            'vit_large_patch16_384',
            'vit_base_patch16_224_in21k'
        ]
        print(version)
        assert version in available_versions, f"version must be one of available_versions"
        self.extractor = timm.create_model(version,
                                            pretrained=from_pretrained,
                                            in_chans=in_channels)
        self.feature_dim = self.extractor.num_features  # num_features for consistency with other models
        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.extractor.forward_features(x)
        # https://github.com/rwightman/pytorch-image-models/blob/8ff45e41f7a6aba4d5fdadee7dc3b7f2733df045/timm/models/vision_transformer.py#L542
        x = x[:, self.extractor.num_prefix_tokens:].mean(dim=1) if self.extractor.global_pool == 'avg' else x[:, 0]
        return x
