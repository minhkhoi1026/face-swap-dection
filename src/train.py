import torch
from pytorch_lightning.trainer import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import datetime

from src.callback import CALLBACK_REGISTRY
from src.model import MODEL_REGISTRY
from src.utils.opt import Opts

def train(config):
    model = MODEL_REGISTRY.get(config["model"]["name"])(config)

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['global']['run_name']}-{time_str}"

    wandb_logger = WandbLogger(
        project=config["global"]["project_name"],
        name=run_name,
        save_dir=config["global"]["save_dir"],
        entity=config["global"]["username"],
    )
    wandb_logger.watch((model))
    
    # only save on rank-0 process if run on multiple GPUs
    # https://github.com/Lightning-AI/lightning/issues/13166
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)

    callbacks = [
        CALLBACK_REGISTRY.get(mcfg["name"])(**mcfg["args"])
        for mcfg in config["callbacks"]
    ]
    print(config["trainer"]["gpus"])
    trainer = pl.Trainer(
        default_root_dir=".",
        max_epochs=config["trainer"]["num_epochs"],
        gpus=config["trainer"]["gpus"] if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=config["trainer"]["evaluate_interval"],
        log_every_n_steps=config["trainer"]["log_interval"],
        enable_checkpointing=True,
        # accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        # sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16 if config["trainer"]["use_fp16"] else 32,
        fast_dev_run=config["trainer"]["debug"], # turn on if you only want to debug
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=-1,  # Sanity full validation required for visualization callbacks
        deterministic=config["trainer"]["deterministic"],
        auto_lr_find=True,
        resume_from_checkpoint=config["global"]["resume"]
    )

    trainer.fit(model)


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    train(cfg)
