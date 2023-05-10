import abc
from src.metrics import METRIC_REGISTRY
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
        self.metric = None
        self.init_model()

    def setup(self, stage):
        if stage in ["fit", "validate", "test"]:
            # generate train and validation pytorch dataset
            # image transform for data augmentation
            image_size = self.cfg["model"]["input_size"]
            image_transform_train = TRANSFORM_REGISTRY.get(
                'train_classify_tf')(img_size=image_size)
            image_transform_val = TRANSFORM_REGISTRY.get('test_classify_tf')(
                img_size=image_size)
            img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()

            self.train_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["name"])(
                img_transform=image_transform_train,
                img_normalize=img_normalize,
                **self.cfg["dataset"]["args"]["train"],
            )

            self.val_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["name"])(
                img_transform=image_transform_val,
                img_normalize=img_normalize,
                **self.cfg["dataset"]["args"]["val"],
            )
            
            self.metric = [
                METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
                for mcfg in self.cfg["metric"]
            ]
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
        self.log("train/loss", loss)
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metric for each batch
        self.log("val/loss", loss)
        for m in self.metric:
            m.update(forwarded_batch, batch)

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        """
        Callback at validation epoch end to do additional works
        with output of validation step, note that this is called
        before `training_epoch_end()`
        Args:
            outputs: output of validation step
        """
        # 1. Calculate average validation loss
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # 2. Calculate metric value
        out = {"val/loss": loss}
        for m in self.metric:
            # 3. Update metric for each batch
            metric_dict = m.value()
            out.update(metric_dict)
            for k in metric_dict.keys():
                self.log(f"val/{k}", out[k])

        # Log string
        log_string = ""
        for metric, score in out.items():
            if isinstance(score, (int, float)):
                log_string += metric + ": " + f"{score:.5f}" + " | "
        log_string += "\n"
        print(log_string)

        # 4. Reset metric
        for m in self.metric:
            m.reset()

        self.log("val/loss", loss.cpu().numpy().item())
        return {**out, "log": out}

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            **self.cfg["data_loader"]["train"]["args"],
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            **self.cfg["data_loader"]["val"]["args"],
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), **self.cfg["optimizer"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.cfg["lr_scheduler"]["args"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler,
                             "monitor": self.cfg["lr_scheduler"]["monitor"], 
                             "interval": "epoch"
                            },
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
