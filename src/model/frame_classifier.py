from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn

from src.extractor import EXTRACTOR_REGISTRY
from src.model.abstract import MLP, AbstractModel
from src.loss.focal_loss import FocalLoss


class FrameClassifier(AbstractModel):
    def init_model(self):
        extractor_cfg = self.cfg["model"]["extractor"]
        self.img_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["name"])(
            **extractor_cfg["params"]
        )
        self.msr_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["name"])(
            **extractor_cfg["params"]
        )
        self.embed_dim = self.cfg["model"]["embed_dim"]

        self.feat_attention = None # TODO
        self.feat_mlp = MLP(
            self.feat_attention,
            self.embed_dim,
            num_hidden_layer=self.cfg["model"]["mlp"]["num_hidden_layer"],
        )
        self.output_layer = nn.Linear(current_reduced_dim, current_reduced_dim // 2)

        self.loss = FocalLoss(num_classes=self.cfg["model"]["num_classes"])
        
    def forward(self, batch):
        img_batch, msr_img_batch = batch["imgs"], batch["msr_imgs"]
        img_feat, msr_img_feat = self.img_extractor(img_batch), self.msr_extractor(msr_img_batch)
        feat = self.feat_attention(torch.stack([img_feat, msr_img_feat], axis=1))
        logits = self.feat_mlp(feat)

        return {
            "logits": logits
        }

    def compute_loss(self, forwarded_batch, input_batch):
        logits, labels = forwarded_batch["logits"], input_batch["labels"]
        return self.loss(logits, labels)
