from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn

from src.extractor import EXTRACTOR_REGISTRY
from src.model.abstract import MLP, AbstractModel
from src.loss.distil_loss import DistillationLoss
from src.loss.focal_loss import FocalLoss


class DistillationFrameClassifier(AbstractModel):
    def init_model(self):
        extractor_cfg = self.cfg["model"]["extractor"]
        self.img_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["img_encoder"]["name"])(
            **extractor_cfg["img_encoder"]["args"]
        )
        self.mlp = nn.Linear(self.img_extractor.feature_dim, self.cfg["model"]["num_classes"])

        self.distil_loss = DistillationLoss(**self.cfg["model"]["teacher_config"])
        self.classfiy_loss = FocalLoss(self.cfg["model"]["num_classes"])
        self.alpha = self.cfg["model"]["alpha"]
        
    def forward(self, batch):
        img_batch = batch["imgs"]
        img_feat = self.img_extractor(img_batch)
        logits = self.mlp(img_feat)

        return {
            "logits": logits
        }

    def compute_loss(self, forwarded_batch, input_batch):
        logits, labels = forwarded_batch["logits"], input_batch["labels"]
        imgs = forwarded_batch["imgs"]
        return self.alpha * self.distil_loss(logits, imgs) + \
                (1 - self.alpha) * self.classfiy_loss(logits, labels)
    
    def extract_target_from_batch(self, batch):
        return batch["labels"].argmax(dim=1)
    
    def extract_pred_from_forwarded_batch(self, forwarded_batch):
        # change to probability since raw logits are passed
        preds = nn.Softmax(dim=1)(forwarded_batch["logits"])
        preds = preds[:, 1]  
        return preds
