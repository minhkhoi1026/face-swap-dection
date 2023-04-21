from pytorch_metric_learning.losses import CrossBatchMemory, NTXentLoss
from sklearn.preprocessing import LabelEncoder
import torch
from src.extractor import EXTRACTOR_REGISTRY
from src.model.abstract import MLP, AbstractModel


class BaselineModel(AbstractModel):
    def init_model(self):
        extractor_cfg = self.cfg["model"]["extractor"]
        self.pc_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["pointcloud"]["name"])(
            **extractor_cfg["pointcloud"]["params"]
        )
        self.lang_extractor = EXTRACTOR_REGISTRY.get(extractor_cfg["text"]["name"])(
            **extractor_cfg["text"]["params"]
        )
        self.embed_dim = self.cfg["model"]["embed_dim"]

        encoder_cfg = self.cfg["model"]["encoder"]
        self.lang_encoder = MLP(
            self.lang_extractor,
            self.embed_dim,
            num_hidden_layer=encoder_cfg["pointcloud"]["num_hidden_layer"],
        )
        self.pc_encoder = MLP(
            self.pc_extractor,
            self.embed_dim,
            num_hidden_layer=encoder_cfg["text"]["num_hidden_layer"],
        )

        self.constrastive_loss = NTXentLoss()
        self.xbm = CrossBatchMemory(
            loss=NTXentLoss(),
            embedding_size=self.embed_dim,
            memory_size=self.cfg["model"]["xbm"]["memory_size"],
        )

    def forward(self, batch):
        pc_embedding_feats = self.pc_encoder.forward(batch["point_clouds"])
        query_embedding_feats = self.lang_encoder.forward(batch["queries"])

        return {
            "pc_embedding_feats": pc_embedding_feats,
            "query_embedding_feats": query_embedding_feats,
        }

    def compute_loss(self, forwarded_batch, input_batch):
        """
        Concatenatae point cloud embedding feature and query embedding feature
        to calculate pair-based loss (here we use InfoNCE loss).
        Label are generated from query_ids (here we consider each query as a "class").
        First we train the model with InfoNCE loss. After certain step, we apply
        Cross Batch Memory method in addiontional with InfoNCE to increase hard mining ability.

        Args:
            forwarded_batch: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """
        emb = torch.cat(
            [
                forwarded_batch["pc_embedding_feats"],
                forwarded_batch["query_embedding_feats"],
            ]
        )  # (batch_size * 2, embed_dim)

        # label is categoricalized id of queries (but repeated 2 time since we concatenated the pc and query)
        labels = torch.tensor(
            LabelEncoder().fit_transform(input_batch["query_ids"]), dtype=torch.int
        ).repeat(
            2
        )  # (batch_size * 2)

        if self.current_epoch >= self.cfg["model"]["xbm"]["enable_epoch"]:
            loss = self.xbm(emb, labels)
        else:
            loss = self.constrastive_loss(emb, labels)
        return loss
