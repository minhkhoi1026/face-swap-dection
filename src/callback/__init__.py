from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from src.utils.registry import Registry

# from src.callback.visualizer_callbacks import VisualizerCallback

CALLBACK_REGISTRY = Registry("CALLBACK")

CALLBACK_REGISTRY.register(EarlyStopping)
CALLBACK_REGISTRY.register(ModelCheckpoint)
CALLBACK_REGISTRY.register(LearningRateMonitor)
# TODO: add WandB visualizer callback
# CALLBACK_REGISTRY.register(VisualizerCallback)
