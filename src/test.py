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

def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return str(m)
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

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
    # import json
    # json.dump(nested_children(model), open("model.json", "w"), indent=4)
    from torchviz import make_dot
    # Create the test dataloader

    # Get a batch of data from the test loader
    # image transform for data augmentation
    from src.augmentation import TRANSFORM_REGISTRY
    from src.dataset import DATASET_REGISTRY
    image_size = cfg["model"]["input_size"]
    image_transform_test = TRANSFORM_REGISTRY.get('test_classify_tf')(
        img_size=image_size)
    img_normalize = TRANSFORM_REGISTRY.get("img_normalize")()

    model.test_dataset = DATASET_REGISTRY.get(cfg["dataset"]["name"])(
        img_transform=image_transform_test,
        img_normalize=img_normalize,
        **cfg["dataset"]["args"]["test"],
    )
            
    # batch = next(iter(model.test_dataloader()))
    # make_dot(model(batch)["logits"], params=dict(model.named_parameters())).render("model", format="svg")
    trainer.test(model)
    del trainer
    del cfg
    del model

if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    check(cfg)
