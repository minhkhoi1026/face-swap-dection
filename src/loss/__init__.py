from src.utils.registry import Registry
LOSS_REGISTRY = Registry("LOSS")

from src.loss.focal_loss import FocalLoss
