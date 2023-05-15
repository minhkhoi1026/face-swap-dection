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


def check(cfg):
    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(cfg['global']['resume'],
                                       cfg=cfg,
                                       strict=True)
    trainer = pl.Trainer(
        gpus=-1
        if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
    )
    trainer.test(model)
    del trainer
    del cfg
    del model

if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    check(cfg)
