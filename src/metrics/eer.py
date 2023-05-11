from typing import List
import torch
from torch import nn
from . import METRIC_REGISTRY
import numpy as np
from sklearn.metrics import roc_curve

@METRIC_REGISTRY.register()
class BinaryEqualErrorRate:
    """
    Equal Error Rate for Binary Classification
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def update(self, preds, targets):
        """
        Perform calculation based on prediction and targets
        """
        preds = preds.detach().cpu().float()
        targets = targets.detach().cpu().float()
        self.preds += preds.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def reset(self):
        self.targets = []
        self.preds = []

    def value(self):
        fpr, tpr, threshold = roc_curve(self.targets, self.preds, pos_label=1)
        fnr = 1 - tpr
        id = np.nanargmin(np.absolute((fnr - fpr)))
        return {f"eer": fpr[id], "threshold": threshold[id]}
