from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn

from src.extractor import EXTRACTOR_REGISTRY
from src.model.abstract import MLP, AbstractModel
from src.loss.focal_loss import FocalLoss


class SingleHeadFrameClassifier(AbstractModel):
    def init_model(self):
        extractor_cfg = self.cfg["model"]["extractor"]
        self.img_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["img_encoder"]["name"])(
            **extractor_cfg["img_encoder"]["args"]
        )
        self.mlp = nn.Linear(self.img_extractor.feature_dim, self.cfg["model"]["num_classes"])

        self.loss = FocalLoss(num_classes=self.cfg["model"]["num_classes"])
        
    def forward(self, batch):
        img_batch = batch["imgs"]
        img_feat = self.img_extractor(img_batch)
        logits = self.mlp(img_feat)

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
