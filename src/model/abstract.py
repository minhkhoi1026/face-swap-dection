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
from torchmetrics import MetricCollection
from src.augmentation import TRANSFORM_REGISTRY
from src.dataset import DATASET_REGISTRY
from src.utils.device import detach

class AbstractModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_metric = None
        self.val_metric = None
        self.test_metric = None
        self.init_model()

    def setup(self, stage):
        if stage in ["fit", "validate"]:
            # generate train and validation pytorch dataset
            # image transform for data augmentation
            image_size = self.cfg["model"]["input_size"]
            image_transform_train = TRANSFORM_REGISTRY.get(
                self.cfg["dataset"]["transform"]["train"])(img_size=image_size)
            image_transform_val = TRANSFORM_REGISTRY.get(self.cfg["dataset"]["transform"]["val"])(
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
            
            metrics = [
                METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
                for mcfg in self.cfg["metric"]
            ]
        elif stage == "test":
            # generate train and validation pytorch dataset
            # image transform for data augmentation
            image_size = self.cfg["model"]["input_size"]
            image_transform_test = TRANSFORM_REGISTRY.get(self.cfg["dataset"]["transform"]["test"])(
                img_size=image_size)
            img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()

            self.test_dataset = DATASET_REGISTRY.get(self.cfg["dataset"]["name"])(
                img_transform=image_transform_test,
                img_normalize=img_normalize,
                **self.cfg["dataset"]["args"]["test"],
            )
            
        metrics = [
            METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
            if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
            for mcfg in self.cfg["metric"]
        ]
        metrics = MetricCollection({metric.name: metric for metric in metrics})
        self.train_metric = metrics.clone(prefix="train/")
        self.val_metric = metrics.clone(prefix="val/")
        self.test_metric = metrics.clone(prefix="test/")
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
    
    @abc.abstractmethod
    def extract_target_from_batch(self, batch):
        pass
    
    @abc.abstractmethod
    def extract_pred_from_forwarded_batch(self, forwarded_batch):
        pass    

    def training_step(self, batch, batch_idx):
        # 1. get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        
        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_batch(forwarded_batch)
        output = self.train_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metric for each batch
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        
        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_batch(forwarded_batch)
        output = self.val_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True)

        return {"loss": detach(loss)}

    def validation_epoch_end(self, outputs) -> None:
        """
        Callback at validation epoch end to do additional works
        with output of validation step, note that this is called
        before `training_epoch_end()`
        Args:
            outputs: output of validation step
        """
        pass
    
    def test_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metric for each batch
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        
        targets = self.extract_target_from_batch(batch)
        preds = self.extract_pred_from_forwarded_batch(forwarded_batch)
        output = self.test_metric(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=True)
        
        # 4. return test data to accumulate and export to csv at end of epoch
        img_paths = batch["img_paths"]
        return {"loss": loss.detach(), 
                "preds": preds.detach().cpu().numpy(), 
                "targets": targets.detach().cpu().numpy(), 
                "img_paths": img_paths}
    
    def test_epoch_end(self, outputs):
        import pandas as pd
        preds = np.concatenate([x["preds"] for x in outputs])
        targets = np.concatenate([x["targets"] for x in outputs])
        img_paths = np.concatenate([x["img_paths"] for x in outputs])
        
        result = pd.DataFrame({"pred": preds, "target": targets, "img_path": img_paths})
        self.logger.log_table("result", dataframe=result)

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
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.test_dataset.collate_fn,
            **self.cfg["data_loader"]["test"]["args"],
        )
        return test_loader

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
