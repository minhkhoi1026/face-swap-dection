"""
Script to test the model
Will use the same config yaml file as train, just need to adjust these below fields:
- dataset.args.test
- data_loader.test
- global.resume: checkpoint path
- metric [optional]: change it whether you want different metrics with training stage
"""

import torch
from src.utils.opt import Opts

from src.model import MODEL_REGISTRY
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import datetime


def check(cfg):
    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(cfg['global']['resume'],
                                       cfg=cfg,
                                       strict=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"eval-{cfg['global']['run_name']}-{time_str}"

    wandb_logger = WandbLogger(
        project=cfg["global"]["project_name"],
        name=run_name,
        save_dir=cfg["global"]["save_dir"],
        entity=cfg["global"]["username"],
    )
    wandb_logger.experiment.config.update(cfg)
    
    trainer = pl.Trainer(
        gpus=-1
        if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        logger=wandb_logger,
    )
    from torchinfo import summary
    print(summary(model))
    trainer.test(model)
    del trainer
    del cfg
    del model

if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    check(cfg)
