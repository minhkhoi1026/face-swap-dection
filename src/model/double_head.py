from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn

from src.extractor import EXTRACTOR_REGISTRY
from src.model.abstract import AbstractModel
from src.loss.focal_loss import FocalLoss
from src.model.feat_attention import FeatAttention

class DoubleHeadFrameClassifier(AbstractModel):
    def init_model(self):
        extractor_cfg = self.cfg["model"]["extractor"]
        self.img_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["img_encoder"]["name"])(
            **extractor_cfg["img_encoder"]["args"]
        )
        self.img_variant_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["img_variant_encoder"]["name"])(
            **extractor_cfg["img_variant_encoder"]["args"]
        )
        embed_dim = self.img_variant_extractor.feature_dim

        self.feat_attention = FeatAttention(embed_dim)
        
        self.mlp = nn.Linear(embed_dim, self.cfg["model"]["num_classes"])

        self.loss = FocalLoss(num_classes=self.cfg["model"]["num_classes"])
        
    def forward(self, batch):
        img_batch, img_variant_batch = batch["imgs"], batch["img_variants"]
        img_feat, img_variant_feat = self.img_extractor(img_batch), self.img_variant_extractor(img_variant_batch)
        feat = self.feat_attention(torch.stack([img_feat, img_variant_feat], axis=1))
        logits = self.mlp(feat)

        return {
            "logits": logits
        }

    def compute_loss(self, forwarded_batch, input_batch):
        logits, labels = forwarded_batch["logits"], input_batch["labels"]
        return self.loss(logits, labels)
    
    def extract_target_from_batch(self, batch):
        return batch["labels"].argmax(dim=1)
    
    def extract_pred_from_forwarded_batch(self, forwarded_batch):
        # change to probability since raw logits are passed
        preds = nn.Softmax(dim=1)(forwarded_batch["logits"])
        preds = preds[:, 1]  
        return preds
