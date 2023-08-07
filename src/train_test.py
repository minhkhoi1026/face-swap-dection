from pytorch_lightning.trainer import seed_everything
from src.utils.opt import Opts
from src.train import train
from src.test import check
import os


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    logid = train(cfg)

    log_path = os.path.join(cfg["global"]["save_dir"], cfg["global"]["project_name"], logid, "checkpoints")
    min_eer = 1
    for file in os.listdir(log_path):
        eer = float(file.split('=')[2].split('-')[0])
        if min_eer >= eer:
            min_eer = eer
            resume = os.path.join(log_path,file)
    print("resume:",resume)
    cfg['global']['resume'] = resume
    check(cfg)
