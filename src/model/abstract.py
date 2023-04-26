import abc
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import random
from src.augmentation import TRANSFORM_REGISTRY
from src.dataset import DATASET_REGISTRY

from src.extractor.base_extractor import ExtractorNetwork

class AbstractModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.metric_evaluator = None
        self.init_model()

    def setup(self, stage):
        if stage in ["fit", "validate", "test"]:
            # generate train and validation pytorch dataset
            # image transform for data augmentation
            image_size = self.cfg["data"]["args"]["SIZE"]
            image_transform_train = TRANSFORM_REGISTRY.get(
                'train_classify_tf')(img_size=image_size)
            image_transform_val = TRANSFORM_REGISTRY.get('test_classify_tf')(
                img_size=image_size)

            self.train_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["train"]["name"])(
                img_transform=image_transform_train,
                **self.cfg["dataset"]["train"]["params"],
            )

            self.val_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["val"]["name"])(
                img_transform=image_transform_val,
                **self.cfg["dataset"]["val"]["params"],
            )

    @abc.abstractmethod
    def init_model(self):
        """
        Function to initialize model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, forwarded_output, input_batch):
        """
        Function to compute loss
        Args:
            forwarded_batch: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update monitor
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metric for each batch
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.metric_evaluator.append(
        #     g_emb=forwarded_batch["pc_embedding_feats"].float().clone().detach(),
        #     q_emb=forwarded_batch["query_embedding_feats"].float().clone().detach(),
        #     query_ids=batch["query_ids"],
        #     gallery_ids=batch["point_cloud_ids"],
        #     target_ids=batch["point_cloud_ids"],
        # )

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        """
        Callback at validation epoch end to do additional works
        with output of validation step, note that this is called
        before `training_epoch_end()`
        Args:
            outputs: output of validation step
        """
        # TODO: add EER evaluate metric
        # self.log_dict(
        #     self.metric_evaluator.evaluate(),
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        # )
        # self.metric_evaluator.reset()
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            **self.cfg["data_loader"]["train"]["params"],
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            **self.cfg["data_loader"]["val"]["params"],
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), **self.cfg["optimizer"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.cfg["lr_scheduler"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, feature_dim, latent_dim=128, num_hidden_layer=2):
        super().__init__()
        self.feature_dim = feature_dim

        layers = []
        current_reduced_dim = self.feature_dim
        for i in range(num_hidden_layer):
            layers.append(nn.Linear(current_reduced_dim, current_reduced_dim // 2))
            layers.append(nn.ReLU())
            current_reduced_dim //= 2

        assert (
            current_reduced_dim >= latent_dim
        ), f"Reduced dim cannot less than embed dim ({current_reduced_dim} < {latent_dim})!"

        layers.append(nn.Linear(current_reduced_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x
