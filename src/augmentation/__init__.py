# https://github.com/kaylode/theseus/blob/master/theseus/base/augmentations/albumentation.py
from src.utils.registry import Registry

TRANSFORM_REGISTRY = Registry('TRANSFORM')

from .default import train_classify_tf, test_classify_tf
from .replay_augment import train_classify_replay_tf
