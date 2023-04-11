from src.utils.registry import Registry
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from wandb.keras import WandbMetricsLogger
import os

CALLBACK_REGISTRY = Registry("CALLBACK")

class CustomModelCheckpoint(ModelCheckpoint):
    """
    Custom model checkpoint to save at specific directory
    """
    def __init__(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        kwargs["filepath"] = os.path.join(save_dir, kwargs["filepath"])
        super().__init__(**kwargs)

# early stopping callback, stop as long as the validation loss does not decrease anymore
CALLBACK_REGISTRY.register(EarlyStopping)
# reduce learning rate callback, decrease learning rate if val loss does not decrease
CALLBACK_REGISTRY.register(ReduceLROnPlateau)
# callback to log metric to wandb
CALLBACK_REGISTRY.register(WandbMetricsLogger)
# callback to log to csv file
CALLBACK_REGISTRY.register(CSVLogger)
# callback to save model checkpoint
CALLBACK_REGISTRY.register(CustomModelCheckpoint)
